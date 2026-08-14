# MasaiCapstoneProject
# Python, Machine Learning & GenAI Projects

This repository contains three modules completed in sequence. The project moves from web data collection, to data analysis and machine learning, and finally to a GenAI/RAG application.

---

## Project Setup

### 1. Install Python

Python 3.11 is recommended.

Check the installation:

```bash
python --version
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```


### 3. Install All Dependencies

From the project root folder:

```bash
pip install -r requirements.txt
```

---

# Module1 – Books Data Scraping & Database

## Summary

This module collects book information from **Books to Scrape**, cleans the data, converts GBP prices to INR, stores the data in SQLite, and performs SQL and Pandas analysis.

### Design Decisions

- **Requests + BeautifulSoup:** used for simple HTML scraping.
- **Pandas:** used for cleaning and transforming the scraped data.
- **Fixed GBP-to-INR rate:** keeps the assignment reproducible.
- **SQLite:** provides a lightweight relational database without requiring a separate database server.
- **Two tables:** `categories` and `books` keep the database structure simple and relational.
- **SQL + Pandas:** the same type of join is demonstrated using both approaches.

The scraper validates that the dataset contains at least 60 books across at least 3 categories.

## End-to-End Run

From the Module1 folder, run the Python scraper file:

```bash
python module1.py
```

The process produces:

```text
all_books_data.csv
cleaned_books_data.csv
final_books_data.csv
books_database.db
```

The final database can then be queried using the SQL queries included in the module.

---

# Module2 – Titanic Data Analysis & Machine Learning

## Summary

This module starts with Exploratory Data Analysis and then uses the cleaned Titanic data to build and compare machine learning models.

### Design Decisions

- **EDA before modeling:** helps understand missing values, outliers, correlations and survival patterns before training models.
- **Pipeline-based preprocessing:** keeps preprocessing together with the model and prevents test-data leakage.
- **Three classification models:** Logistic Regression, Decision Tree and Random Forest provide a simple model comparison.
- **SMOTE and class weights:** used to study different approaches to class imbalance.
- **GridSearchCV:** used to tune the Random Forest model.
- **Joblib:** used to save the complete best model pipeline for reuse.

## End-to-End Run

Run the EDA script first:

```bash
python 01_eda.py
```

This creates:

```text
titanic.csv
```

Then run the modeling script:

```bash
python 02_modeling.py
```

The modeling step produces files such as:

```text
titanic_best_pipeline.joblib
classification_results.csv
imbalance_results.csv
regression_results.csv
```

It also creates PNG files containing charts and model visualizations.

---

# Module3 – Zepto GenAI RAG Service

## Summary

This module is a small RAG-based question-answering service for Zepto policy questions. It uses policy documents, local embeddings, ChromaDB retrieval, LangGraph workflow, Pydantic validation and FastAPI.

### Design Decisions

- **Local `all-MiniLM-L6-v2` embeddings:** keeps the required mock workflow independent of an external LLM.
- **ChromaDB:** provides vector storage and similarity-based retrieval.
- **LangGraph:** separates intent classification, retrieval/answering and direct answers into clear workflow nodes.
- **Mock mode by default:** makes the application deterministic and avoids requiring an API key.
- **Pydantic:** ensures a consistent response structure containing `answer`, `sources` and `confidence`.
- **FastAPI:** provides a simple API endpoint that can be tested through Swagger.

## End-to-End Run

From the Module3 folder:

```bash
python module3.py
```

This indexes the policy documents and runs example queries.

Then start the API:

```bash
uvicorn main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

Select:

```text
POST /ask
```

Example request:

```json
{
  "query": "What is the delivery fee below INR 149?"
}
```

You can also test a general question:

```json
{
  "query": "What is the capital of France?"
}
```

The application should return a structured JSON response with the answer, sources and confidence.

### Optional Real LLM Mode

The project also supports an optional Groq-based LLM mode.

**Windows PowerShell:**

```powershell
$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your_api_key"
uvicorn main:app --reload
```

Do not hardcode the API key or commit it to GitHub.

---

# Overall Project Flow

```text
Module1
Web Scraping
    ↓
Data Cleaning & Database
    ↓
Module2
EDA & Machine Learning
    ↓
Model Evaluation & Saving
    ↓
Module3
Embeddings & Retrieval
    ↓
RAG & FastAPI
```

Together, the three modules demonstrate a progression from **raw data collection → data analysis → machine learning → GenAI application development**.
