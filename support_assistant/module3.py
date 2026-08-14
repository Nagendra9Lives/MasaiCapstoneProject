"""
Zepto GenAI RAG Service - Consolidated Version

This single file implements the complete required Python application:
1. Eight Zepto policy documents
2. Document ingestion/chunking
3. Sentence Transformer embeddings
4. ChromaDB vector storage
5. LangGraph StateGraph with 3 nodes
6. MOCK_LLM baseline
7. Optional real LLM path
8. Structured Pydantic response
9. FastAPI POST /ask endpoint

Run:
    pip install -r requirements.txt
    python main.py

Then in another terminal:
    uvicorn main:app --reload

Default:
    MOCK_LLM=1

Optional real LLM:
    MOCK_LLM=0
    GROQ_API_KEY=your_key
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "zepto_policies"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)


# ============================================================
# 2. ZEpto POLICY CORPUS
# ============================================================

DOCUMENTS = {
    "doc_01": """
Zepto delivers grocery and household essentials to serviceable pin codes
within 10 to 30 minutes of order confirmation, depending on the customer's
delivery zone and current order volume. Standard delivery is free on orders
over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Priority delivery, which reserves the next available rider slot, is available
at checkout for an additional INR 15. Zepto does not currently deliver to
addresses outside its listed serviceable pin codes.
""".strip(),

    "doc_02": """
Grocery and perishable items may be reported for a return within 24 hours of
delivery if damaged, spoiled, or incorrect; non-perishable packaged items may
be returned within 7 days of delivery in unopened, resalable condition.
Approved refunds are credited to the original payment method within 3–5
business days, or instantly to the Zepto wallet if the customer opts for
wallet credit. Personal care items that have been opened are non-returnable
except in the case of a manufacturing defect. Return pickup, where required,
is arranged free of cost by Zepto.
""".strip(),

    "doc_03": """
Zepto offers three account tiers: Basic (free, default tier, standard
delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery
on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per
month, free priority delivery, 10% off select categories, and early access
to limited-time deals 24 hours before they go live to Basic and Pass
members). Membership can be cancelled at any time from account settings;
cancelling stops the next billing cycle but does not refund the current
membership period.
""".strip(),

    "doc_04": """
Every Zepto order shows a live rider-tracking map from the moment it is
packed until delivery, accessible from the 'Track Order' screen. Estimated
delivery time updates automatically as the rider moves. If an order's status
shows no movement for more than 20 minutes past its original estimated
delivery time, customers should contact support directly rather than
continue waiting, since this indicates a likely delivery issue.
""".strip(),

    "doc_05": """
Orders can be cancelled free of cost any time before the order status changes
to 'Packed', typically within the first 2 minutes of placing the order. Once
an order has been packed, it can no longer be cancelled through the app,
since the rider is dispatched immediately after packing given Zepto's
quick-delivery model. If a packed order cannot be delivered due to a
Zepto-side issue (for example, rider unavailability), the order is
auto-cancelled and fully refunded without any cancellation fee.
""".strip(),

    "doc_06": """
If an order arrives with damaged, spoiled, or missing items, customers must
report it within 24 hours of delivery through the 'Report an Issue' button on
the order page. Zepto ships a free replacement or issues a full refund for
damaged, spoiled, or missing items without requiring the customer to return
the original item, unless the order value exceeds INR 1000, in which case a
photo of the issue must be submitted through the report form before a
replacement or refund is processed.
""".strip(),

    "doc_07": """
Zepto gift cards are available in fixed denominations of INR 100, INR 250,
INR 500, and INR 1000, and are delivered by email or SMS within minutes of
purchase. Gift cards are valid for 1 year from the date of issue and carry
no maintenance fees. Gift card balance can be combined with one other
payment method at checkout but cannot be combined with another gift card
in the same transaction. Gift card balance cannot be redeemed for cash
except where required by law.
""".strip(),

    "doc_08": """
Zepto customer support is available via in-app chat 24 hours a day,
7 days a week, given the time-sensitive nature of quick commerce deliveries.
Average in-app chat response time is under 2 minutes. Email support is also
available for non-urgent queries and is answered within 24 hours on business
days. Phone support is not offered.
""".strip(),
}


# ============================================================
# 3. STRUCTURED PROMPT
# ============================================================

RAG_PROMPT = """
ROLE:
You are a Zepto policy assistant. Answer questions using only the supplied
Zepto policy context.

CONTEXT:
The application retrieves the most relevant Zepto policy chunks from ChromaDB.
The retrieved chunks are the only trusted source for policy facts.

TASK:
Answer the user's question clearly and directly using the retrieved context.

FORMAT:
Return JSON with exactly these fields:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise, normally 1 to 4 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, dates, limits, or support procedures.

FEW-SHOT EXAMPLE:
Question:
How much is the standard delivery fee for an order below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee.

Good JSON:
{
  "answer": "Orders below INR 149 have a flat INR 25 standard delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}
"""


# ============================================================
# 4. PYDANTIC MODELS
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# 5. LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]
    response: AnswerResponse


# ============================================================
# 6. EMBEDDING MODEL
# ============================================================

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is None:
        print("Loading embedding model:", MODEL_NAME)
        _embedder = SentenceTransformer(MODEL_NAME)

    return _embedder


# ============================================================
# 7. CHROMADB
# ============================================================

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def ingest_documents():
    """
    Load the 8 documents.

    For this assignment, every short document is treated as one chunk.
    This satisfies the allowed simple per-document chunking approach.
    """

    collection = get_collection()
    embedder = get_embedder()

    ids = list(DOCUMENTS.keys())
    texts = list(DOCUMENTS.values())

    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
    ).tolist()

    metadatas = [
        {
            "document_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_01",
        }
        for doc_id in ids
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"ChromaDB collection '{COLLECTION_NAME}' "
        f"contains {collection.count()} chunks."
    )

    return collection.count()


# ============================================================
# 8. RETRIEVAL
# ============================================================

def retrieve(query: str, top_k: int = 3):
    """
    Embed the query and retrieve top-k chunks using cosine similarity.
    Retrieval happens in both mock and real-LLM modes.
    """

    collection = get_collection()

    if collection.count() == 0:
        ingest_documents()

    embedder = get_embedder()

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    for i in range(len(documents)):
        metadata = metadatas[i] if i < len(metadatas) else {}

        retrieved.append(
            {
                "id": ids[i],
                "document_id": metadata.get(
                    "document_id",
                    ids[i],
                ),
                "document": documents[i],
                "distance": distances[i],
            }
        )

    return retrieved


# ============================================================
# 9. MOCK INTENT CLASSIFICATION
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_with_mock(query: str):
    query_lower = query.lower()

    for keyword in POLICY_KEYWORDS:
        if keyword in query_lower:
            return "policy_question"

    return "general_question"


# ============================================================
# 10. OPTIONAL REAL LLM CLIENT
# ============================================================

def get_groq_client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY."
        )

    return Groq(api_key=api_key)


def classify_with_real_llm(query: str):
    client = get_groq_client()

    system_prompt = """
Classify the user's query as exactly one of:

policy_question
general_question

A policy_question asks about Zepto delivery, returns, refunds,
membership, tracking, cancellation, gift cards, or customer
support policies.

Return only the classification.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    answer = response.choices[0].message.content.strip()

    if "policy_question" in answer:
        return "policy_question"

    return "general_question"


# ============================================================
# 11. LANGGRAPH NODE 1 - CLASSIFY INTENT
# ============================================================

def classify_intent(state: RAGState):

    query = state["query"]

    if MOCK_LLM:
        intent = classify_with_mock(query)
    else:
        intent = classify_with_real_llm(query)

    print(
        f"[classify_intent] query={query!r} "
        f"intent={intent}"
    )

    return {
        "intent": intent
    }


# ============================================================
# 12. LANGGRAPH NODE 2 - RETRIEVE AND ANSWER
# ============================================================

def create_mock_answer(retrieved):
    if not retrieved:
        return (
            "Based on the retrieved context: "
            "No matching policy context was found."
        )

    top_chunk = retrieved[0]["document"]

    # Assignment asks for a short excerpt, approximately 200 chars.
    snippet = top_chunk[:200].strip()

    return (
        "Based on the retrieved context: "
        + snippet
    )


def generate_real_answer(
    query: str,
    retrieved: list[dict],
):
    client = get_groq_client()

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[{item['document_id']}]\n"
            f"{item['document']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": RAG_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Return JSON only."
            ),
        },
    ]

    last_error = None

    # Initial attempt + 2 additional retries = maximum 3 attempts.
    for attempt in range(3):

        try:

            current_messages = list(messages)

            if attempt > 0:
                current_messages.append(
                    {
                        "role": "user",
                        "content": """
The previous response failed schema validation.

Return valid JSON only with exactly:
answer: string
sources: list of strings
confidence: number from 0 to 1

Do not add markdown or explanation outside the JSON.
""",
                    }
                )

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                response_format={
                    "type": "json_object"
                },
                messages=current_messages,
            )

            raw = response.choices[0].message.content

            data = json.loads(raw)

            return AnswerResponse.model_validate(data)

        except Exception as error:
            last_error = error
            print(
                f"Real LLM validation attempt "
                f"{attempt + 1} failed: {error}"
            )

    return AnswerResponse(
        answer=(
            "ERROR: The real LLM response could not "
            "be validated after 3 attempts."
        ),
        sources=[],
        confidence=0.0,
    )


def retrieve_and_answer(state: RAGState):

    query = state["query"]

    retrieved = retrieve(
        query=query,
        top_k=3,
    )

    source_ids = [
        item["document_id"]
        for item in retrieved
    ]

    if MOCK_LLM:

        answer = create_mock_answer(
            retrieved
        )

        response = AnswerResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0,
        )

    else:

        response = generate_real_answer(
            query=query,
            retrieved=retrieved,
        )

    print(
        "[retrieve_and_answer] sources:",
        source_ids,
    )

    return {
        "retrieved": retrieved,
        "response": response,
    }


# ============================================================
# 13. LANGGRAPH NODE 3 - DIRECT ANSWER
# ============================================================

def direct_answer(state: RAGState):

    query = state["query"]

    if MOCK_LLM:

        response = AnswerResponse(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0,
        )

    else:

        client = get_groq_client()

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """
Answer the user's general question directly.

Do not pretend that you retrieved Zepto policy information.
Keep the answer concise.
""",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        response = AnswerResponse(
            answer=(
                response
                .choices[0]
                .message
                .content
                .strip()
            ),
            sources=[],
            confidence=0.8,
        )

    print("[direct_answer] general question")

    return {
        "response": response
    }


# ============================================================
# 14. CONDITIONAL ROUTER
# ============================================================

def route_after_classification(
    state: RAGState
):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# 15. BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(RAGState)

    # Required three nodes.
    graph.add_node(
        "classify_intent",
        classify_intent,
    )

    graph.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    graph.add_node(
        "direct_answer",
        direct_answer,
    )

    # Entry point.
    graph.set_entry_point(
        "classify_intent"
    )

    # Required conditional edge.
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer":
                "retrieve_and_answer",

            "direct_answer":
                "direct_answer",
        },
    )

    graph.add_edge(
        "retrieve_and_answer",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    return graph.compile()


# ============================================================
# 16. ASK FUNCTION
# ============================================================

def ask(query: str):

    graph = build_graph()

    result = graph.invoke(
        {
            "query": query
        }
    )

    response = result["response"]

    return AnswerResponse.model_validate(
        response
    )


# ============================================================
# 17. FASTAPI APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting Zepto GenAI RAG Service")
    print("=" * 60)

    print(
        "MOCK_LLM:",
        "1 (required mock mode)"
        if MOCK_LLM
        else "0 (optional real LLM mode)"
    )

    print(
        "Embedding model:",
        MODEL_NAME
    )

    # Ingest local documents.
    # This does not call an LLM.
    ingest_documents()

    print("=" * 60)

    yield


app = FastAPI(
    title="Zepto GenAI RAG Service",
    description=(
        "Zepto policy RAG service using "
        "LangGraph, ChromaDB and FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 18. FASTAPI ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Zepto GenAI RAG API is running",
        "endpoint": "POST /ask",
        "mock_llm": MOCK_LLM,
    }


@app.get("/health")
def health():

    collection = get_collection()

    return {
        "status": "ok",
        "documents": collection.count(),
        "mock_llm": MOCK_LLM,
    }


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    request: AskRequest
):

    return ask(
        request.query
    )


# ============================================================
# 19. LOCAL COMMAND-LINE TESTS
# ============================================================

def run_examples():

    print("\n")
    print("=" * 70)
    print("EXAMPLE 1 - POLICY QUESTION")
    print("=" * 70)

    query_1 = (
        "What is the delivery fee for orders "
        "below INR 149?"
    )

    result_1 = ask(query_1)

    print(
        json.dumps(
            result_1.model_dump(),
            indent=2,
        )
    )

    print("\n")
    print("=" * 70)
    print("EXAMPLE 2 - GENERAL QUESTION")
    print("=" * 70)

    query_2 = (
        "What is the capital of France?"
    )

    result_2 = ask(query_2)

    print(
        json.dumps(
            result_2.model_dump(),
            indent=2,
        )
    )


# ============================================================
# 20. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nZepto GenAI RAG - consolidated application"
    )

    ingest_documents()

    run_examples()

    print("\nTo start the API, run:")
    print(
        "uvicorn main:app --reload"
    )
