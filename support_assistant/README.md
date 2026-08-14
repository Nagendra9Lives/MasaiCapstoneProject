# Zepto GenAI RAG Service

This project is a small GenAI/RAG service built for the Zepto policy question-answering requirement.

It uses 8 Zepto policy documents, local Sentence Transformer embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

### Ingestion

The application contains 8 Zepto policy documents:

1. Delivery Policy
2. Returns & Refunds
3. Membership Tiers
4. Order Tracking
5. Order Cancellation
6. Damaged or Missing Items
7. Gift Cards
8. Customer Support Hours

In the consolidated version, these documents are stored directly inside `main.py` in the `DOCUMENTS` dictionary.

Because each document is short, the implementation uses one chunk per document.

### Embedding

The application uses:

```text
all-MiniLM-L6-v2
```

from `sentence-transformers`.

Each policy document is converted into a vector embedding.

### Vector Storage

The embeddings are stored in a ChromaDB collection named:

```text
zepto_policies
```

Cosine similarity is used for retrieval.

### Retrieval

For a policy question:

1. The query is embedded.
2. ChromaDB compares the query vector with the stored vectors.
3. The top 3 most similar chunks are retrieved.
4. Their document IDs are included in the response.

Retrieval works in both mock and real-LLM modes.

### Generation

In the required mock mode, no LLM is called.

The policy answer follows:

```text
Based on the retrieved context: ...
```

A general question returns:

```text
I can only answer questions about Zepto policies right now.
```

## LangGraph

The application uses a LangGraph `StateGraph` with three required nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

### `classify_intent`

In mock mode, the following keywords are checked:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If a keyword occurs in the lowercased query, it becomes:

```text
policy_question
```

Otherwise:

```text
general_question
```

### `retrieve_and_answer`

This node embeds the query, retrieves the top 3 ChromaDB chunks, and generates the answer.

### `direct_answer`

This node handles general questions and does not retrieve policy documents.

## 4. MOCK_LLM

The graded baseline is the offline mock mode.

The application defaults to:

```text
MOCK_LLM=1
```

If `MOCK_LLM` is not set, mock mode is used.

In mock mode:

- No LLM API is called.
- No API key is required.
- Intent classification uses the required keyword rules.
- Embeddings are generated locally.
- ChromaDB retrieval is real.
- Answers are deterministic.
- Pydantic validation is deterministic.

This should be tested first.

## 5. Structured Prompt

The prompt follows the required:

```text
ROLE
CONTEXT
TASK
FORMAT
LENGTH
```

It also contains:

- an explicit negative constraint
- a few-shot example
- JSON output instructions

The prompt is used by the optional real-LLM path.

## 6. Pydantic Response

The API response is:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
```

The response model contains:

```python
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

`confidence` must be between `0.0` and `1.0`.

For a policy question, `sources` contains retrieved document IDs.

For a general question:

```json
"sources": []
```

## 7. FastAPI

The main endpoint is:

```text
POST /ask
```

Request:

```json
{
  "query": "What is the delivery fee below INR 149?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01", "doc_03", "doc_05"],
  "confidence": 1.0
}
```

The application also has:

```text
GET /
GET /health
```

## 8. Installation

I recommend Python 3.11.

Check:

```bash
python --version
```

Install the required packages:

```bash
pip install fastapi uvicorn pydantic langgraph chromadb sentence-transformers groq
```

The first run may download the `all-MiniLM-L6-v2` model.

## 9. Run the Application

Run:

```bash
python main.py
```

This will index the 8 documents and run two example queries.

Then start FastAPI:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 10. Example 1 - Policy Question

Request:

```json
{
  "query": "What is the delivery fee for orders below INR 149?"
}
```

A typical mock response is:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_05"
  ],
  "confidence": 1.0
}
```

The exact source ordering can vary because retrieval is based on embedding similarity.

## 11. Example 2 - General Question

Request:

```json
{
  "query": "What is the capital of France?"
}
```


Mock response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## 12. Test With Swagger

Start:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Then:

1. Select `POST /ask`.
2. Click `Try it out`.
3. Enter:

```json
{
  "query": "How much is Zepto delivery below INR 149?"
}
```

4. Click `Execute`.
5. Check the JSON response.

Then test:

```json
{
  "query": "What is the capital of France?"
}
```

## 13. Test With curl

### Policy question

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee below INR 149?\"}"
```

### General question

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of France?\"}"
```

## 14. Optional Real LLM Mode

The real LLM path is optional.

To use it:

```text
MOCK_LLM=0
```

The code is prepared for Groq.

Set the API key as an environment variable.

### Windows PowerShell

```powershell
$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your_api_key"
uvicorn main:app --reload
```

### Linux/Mac

```bash
export MOCK_LLM=0
export GROQ_API_KEY="your_api_key"
uvicorn main:app --reload
```

Do not hardcode the API key in `main.py` or commit it to GitHub.

## 15. Real LLM Validation Retry

The optional real-LLM path expects:

```text
answer
sources
confidence
```

If validation fails, the code retries up to two additional times.


After three attempts, the application returns a clearly marked error response.

## 16. Docker

Build:

```bash
docker build -t zepto-genai-rag .
```

Run:

```bash
docker run --rm -p 7860:7860 zepto-genai-rag
```

Open:

```text
http://localhost:7860/docs
```

The Docker container uses mock mode by default.

## 17. Docker Test

After starting the container:

```bash
curl -X POST "http://localhost:7860/ask" -H "Content-Type: application/json" -d "{\"query\":\"How long does Zepto delivery take?\"}"
```

## 18. Consolidated Project Structure

```text
zepto-genai-rag/
│
├── main.py
├── README.md
├── Dockerfile
├── requirements.txt
└── .gitignore
```

The 8 policy documents are included directly in `main.py` in the consolidated version.

## 19. Main Code Sections

The consolidated `main.py` is organized into:

```text
1. Configuration
2. Zepto Policy Corpus
3. Structured Prompt
4. Pydantic Models
5. LangGraph State
6. Embedding Model
7. ChromaDB
8. Retrieval
9. Mock Intent Classification
10. Optional Real LLM Client
11. classify_intent
12. retrieve_and_answer
13. direct_answer
14. Conditional Router
15. Build LangGraph
16. ask()
17. FastAPI Application
18. FastAPI Routes
19. Local Examples
20. Main
```

This makes the single-file implementation easier to understand and submit.

##  Final Run

The simplest setup is:

```bash
pip install fastapi uvicorn pydantic langgraph chromadb sentence-transformers groq
```

Then:

```bash
python main.py
```

Then:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Test:

```json
{
  "query": "What is the delivery fee below INR 149?"
}
```
