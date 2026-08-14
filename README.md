# MasaiCapstoneProject
This repository contains three projects completed in sequence, starting with data collection and analysis and moving toward machine learning and GenAI.

## Module1 – Books Data Scraping & Database
This module focuses on collecting and preparing real web data.

- Scraped book data from **Books to Scrape** using Requests and BeautifulSoup.
- Cleaned the data and converted GBP prices to INR.
- Saved data to CSV files.
- Created a SQLite database with `categories` and `books` tables.
- Used SQL queries and compared SQL results with Pandas.

## Module2 – Titanic Data Analysis & Machine Learning
This module focuses on Exploratory Data Analysis and machine learning.

- Performed EDA on the Titanic dataset.
- Checked missing values, outliers, correlations and survival patterns.
- Prepared the data using pipelines, imputation and one-hot encoding.
- Built Logistic Regression, Decision Tree and Random Forest models.
- Compared model performance and handled class imbalance using SMOTE.
- Tuned Random Forest and saved the best pipeline.
- Also used Linear Regression for fare prediction.

## Module3 – Zepto GenAI RAG Service
This module introduces a small GenAI/RAG application for Zepto policy questions.

- Used 8 Zepto policy documents as the knowledge base.
- Created embeddings using `all-MiniLM-L6-v2`.
- Stored and searched embeddings with ChromaDB.
- Used LangGraph for the question-answering workflow.
- Used Pydantic for structured responses.
- Built a FastAPI `POST /ask` endpoint.
- Included a deterministic mock mode and an optional real LLM mode.

**Main learning:** Documents → embeddings → retrieval → answer generation → API.

## Overall Project Flow

**Module1:** Collect and structure data  
 
**Module2:** Analyze data and build ML models  

**Module3:** Build a GenAI/RAG application

These three modules provide practical experience across **Python, data analysis, databases, machine learning, and GenAI**.
