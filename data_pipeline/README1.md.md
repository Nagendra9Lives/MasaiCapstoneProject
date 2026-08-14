# Books Data Scraping and Database Project

## About This Project

This is a Masai Module1 Python data project:

- Web scraping
- Data cleaning
- Currency conversion
- Working with CSV files
- SQLite database
- SQL queries
- Pandas
- Comparing SQL and Pandas results

The data is collected from **Books to Scrape**, a website made for practicing web scraping.

Website used:

`http://books.toscrape.com/`

---

## Technologies Used

The main Python libraries and tools used in this project are:

- Python
- Requests
- BeautifulSoup
- Pandas
- CSV
- SQLite
- SQL

---

## Python Libraries

The project uses the following imports:

```python
import csv
import sqlite3
import pandas as pd
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
```

### Why these libraries are used

| Library | Purpose |
|---|---|
| `requests` | To download web pages |
| `BeautifulSoup` | To read and extract information from HTML |
| `pandas` | To clean and work with the data |
| `sqlite3` | To create and work with the SQLite database |
| `csv` | To create CSV files |
| `urljoin` | To create complete URLs from relative URLs |

---

## Step 1: Web Scraping

The scraper first opens the Books to Scrape website and finds the available book categories.

For each category, it goes through the available pages and collects book information.

The project collects:

- Category
- Book title
- Price
- Availability
- Rating

The scraper uses `BeautifulSoup` selectors to find the book information from the HTML page.

The scraping code also checks the number of books and categories collected.

The current code sets:

```python
MAX_BOOKS = 100
```

So the scraper stops after collecting up to 100 books.

The validation checks that the dataset contains at least 60 books and at least 3 categories.

---

## Step 2: Save Scraped Data

The scraped data is saved into:

```text
all_books_data.csv
```

The CSV file contains these columns:

```text
category
title
price
availability
rating
```

---

## Step 3: Data Cleaning

After scraping, the CSV file is loaded using Pandas.

The price contains the `£` symbol, so it is removed and the price is converted into a numeric value.

A new column is created:

```text
price_GBP
```

If a price is missing or cannot be converted, the project uses the median price to fill the missing value.

The availability information is also converted into a Boolean column:

```text
in_stock
```

This makes it easier to identify whether a book is currently in stock.

The cleaned data is saved as:

```text
cleaned_books_data.csv
```

---

## Step 4: Currency Conversion

The project converts the book price from GBP to INR.

The fixed conversion rate used in the code is:

```python
GBP_TO_INR_RATE = 105.50
```

The INR price is calculated using:

```text
price_INR = price_GBP × GBP_TO_INR_RATE
```

The final dataset is saved as:

```text
final_books_data.csv
```

---

## Step 5: SQLite Database

The project creates a SQLite database:

```text
books_database.db
```

Two tables are created:

### categories

This table stores the book categories.

Columns:

- `category_id`
- `category_name`

### books

This table stores the book information.

Columns:

- `books_id`
- `title`
- `price_GBP`
- `price_INR`
- `rating`
- `in_stock`
- `category_id`

The `category_id` is used to connect the `books` table with the `categories` table.

This is a simple example of a relational database design.

---

## Step 6: SQL Queries

The project demonstrates five basic SQL queries.

### Query 1 - SELECT and LIMIT

Gets the first five books.

```sql
SELECT title, price_GBP
FROM books
LIMIT 5;
```

### Query 2 - WHERE and ORDER BY

Finds books that are in stock and cost more than £10, then sorts them by price.

```sql
SELECT title, price_GBP
FROM books
WHERE in_stock = 1 AND price_GBP > 10.00
ORDER BY price_GBP DESC;
```

### Query 3 - DISTINCT and BETWEEN

Finds unique ratings for books priced between £5 and £15.

```sql
SELECT DISTINCT rating
FROM books
WHERE price_GBP BETWEEN 5.00 AND 15.00;
```

### Query 4 - IN

Finds books belonging to selected category IDs.

```sql
SELECT title, category_id
FROM books
WHERE category_id IN (1, 3, 5);
```

### Query 5 - JOIN

Combines the books and categories tables and displays highly rated books with their category names.

```sql
SELECT b.title, c.category_name, b.rating
FROM books b
JOIN categories c
ON b.category_id = c.category_id
ORDER BY b.rating DESC
LIMIT 10;
```

---

## Step 7: SQL and Pandas Comparison

One interesting part of this project is that the final JOIN operation is performed in two ways.

### Using SQL

The project uses:

```sql
JOIN
```

to combine the `books` and `categories` tables.

### Using Pandas

The same type of operation is reproduced using:

```python
pd.merge()
```

The results from SQL and Pandas are then placed side-by-side for comparison.

This helped me understand that similar data operations can be performed using both SQL and Pandas.

---

## How to Run the Project

### 1. Install Python

Make sure Python is installed on your computer.

You can check it using:

```bash
python --version
```

### 2. Install Required Libraries

Run:

```bash
pip install requests beautifulsoup4 pandas
```

### 3. Run the Python File

Run the project using:

```bash
python your_file_name.py
```

Replace `your_file_name.py` with the actual Python file name.


---

## Notes

The scraper also includes validation to make sure the final dataset has at least 60 books across at least 3 categories.

---

## Final Result

This project helped me understand how raw web data can be converted into structured data that can be used for analysis and database queries.
