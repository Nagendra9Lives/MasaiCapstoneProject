import csv
import sqlite3
import pandas as pd
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_categories():
  response = requests.get(BASE_URL)
  response.raise_for_status()
  soup = BeautifulSoup(response.text, "html.parser")
  category_tags = soup.select(".nav-list > li > ul > li > a")
  categories = []
  for tag in category_tags:
    cat_name = tag.text.strip()
    cat_url = urljoin(BASE_URL, tag["href"])
    categories.append({"name": cat_name, "url": cat_url})
  print(f"Found {len(categories)} categories.") # Debugging
  return categories

def scrape_category_books(category_url):
  books = []
  current_page_url = category_url
  page_num = 1 # Added for debugging
  while current_page_url:
    print(f"  Scraping page {page_num} from: {current_page_url}") # Debugging
    response = requests.get(current_page_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select("article.product_pod")
    print(f"  Found {len(articles)} articles on page {page_num}.") # Debugging
    if not articles and page_num == 1:
        print(f"  WARNING: No articles found on the first page of {category_url}. Selector 'article.product_pod' might be incorrect or page is empty.")
    for article in articles:
      title_tag = article.select_one("h3 > a")
      title = title_tag["title"] if title_tag else "Unknown"
      price_tag = article.select_one(".price_color")
      price = price_tag.text if price_tag else "N/A"
      avail_tag = article.select_one(".availability")
      availability = avail_tag.text.strip() if avail_tag else "Unknown"
      rating_tag = article.select_one(".star-rating")
      rating = "Unknown"
      if rating_tag:
        classes = rating_tag.get("class", [])
        for cls in classes:
          if cls in RATING_MAP:
            rating = RATING_MAP[cls]
            break
      books.append(
        {
          "title": title,
          "price": price,
          "availability": availability,
          "rating": rating,
        }
      )
    next_button = soup.select_one("li.next > a")
    if next_button:
      current_page_url = urljoin(current_page_url, next_button["href"])
      page_num += 1
    else:
      current_page_url = None
  return books

def main():
  categories = get_categories()
  all_data = []
  book_count = 0 # Initialize book counter
  MAX_BOOKS = 100 # Define maximum books to scrape

  for cat in categories:
    if book_count >= MAX_BOOKS:
        break # Stop if enough books have been scraped

    print(f"Scraping category: {cat['name']}")
    cat_books = scrape_category_books(cat["url"])
    print(f"  Scraped {len(cat_books)} books from category: {cat['name']}") # Debugging
    for book in cat_books:
      if book_count >= MAX_BOOKS:
        break # Stop if enough books have been scraped
      book["category"] = cat["name"]
      all_data.append(book)
      book_count += 1

  # -----------------------------------------
  # Validate scraping requirements (Moved outside the loop)
  # -----------------------------------------

  total_books = len(all_data)
  unique_categories = len(
      set(book["category"] for book in all_data)
  )

  print(f"\nTotal books scraped: {total_books}")
  print(f"Total categories scraped: {unique_categories}")

  # Requirement: >= 60 books and >= 3 categories
  # Adjusting assertion for reduced book count
  assert total_books >= min(MAX_BOOKS, 60), \
      f"ERROR: Less than {min(MAX_BOOKS, 60)} books were scraped."

  assert unique_categories >= 3, \
      "ERROR: Less than 3 categories were scraped."

  print("Scraping validation PASSED!")
  print(
      f"Dataset contains {total_books} books "
      f"across {unique_categories} categories."
  )

  filename = "all_books_data.csv"
  keys = ["category", "title", "price", "availability", "rating"]
  with open(filename, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=keys)
    writer.writeheader()
    writer.writerows(all_data)
  print(f"Scraping complete. Total books scraped: {len(all_data)}. Data saved to {filename}")

if __name__ == "__main__":
  main()


# Load the previously scraped data
df = pd.read_csv("all_books_data.csv")

# 1. Clean and convert the 'price' field
# Strip the currency symbol and convert to float
df["price_GBP"] = df["price"].str.replace("£", "", regex=False)
df["price_GBP"] = pd.to_numeric(df["price_GBP"], errors="coerce")

# Handle missing or corrupted price values using median imputation
# to avoid dropping the row and keep the data intact
median_price = df["price_GBP"].median()
df["price_GBP"].fillna(median_price, inplace=True)

# 2. Convert text ratings to integers
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating"] = df["rating"].map(rating_map)

# Drop rows where the rating could not be mapped to a valid integer
df.dropna(subset=["rating"], inplace=True)
df["rating"] = df["rating"].astype(int)

# 3. Parse availability into a boolean column
df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)

# Select and save the cleaned columns
cleaned_df = df[["category", "title", "price_GBP", "rating", "in_stock"]]
cleaned_df.to_csv("cleaned_books_data.csv", index=False)

print("Data cleaning complete")


# 3. Load the previously cleaned data
df = pd.read_csv("cleaned_books_data.csv")

# Define the fixed conversion rate to INR
GBP_TO_INR_RATE = 105.50 # Replace with the actual baseline fixed rate

# Convert the price to INR
df["price_INR"] = df["price_GBP"] * GBP_TO_INR_RATE

# Save the final dataset
df.to_csv("final_books_data.csv", index=False)

print("Currency conversion complete")


# Connect to the SQLite database
conn = sqlite3.connect('books_database.db')
cursor = conn.cursor()

# 4. Define and execute SQL schema
sql_schema = """
CREATE TABLE IF NOT EXISTS categories (
  category_id INTEGER PRIMARY KEY,
  category_name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS books (
  books_id INTEGER PRIMARY KEY,
  title TEXT,
  price_GBP REAL,
  price_INR REAL,
  rating INTEGER,
  in_stock INTEGER,
  category_id INTEGER,
  FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""
cursor.executescript(sql_schema)
conn.commit()
print("SQL tables created/verified.")

# 1. Insert cleaned data into the tables
# Assuming you have your data in pandas DataFrames named 'categories_df' and 'books_df'
# For demonstration, let's create dummy dataframes or load from CSV if available

# To insert data, we need to process the 'cleaned_df' and 'df' (final_books_data) into suitable formats
# Let's assume 'cleaned_df' is available from previous steps
# First, populate categories table
# Create a unique list of categories with IDs
categories_unique = cleaned_df[['category']].drop_duplicates().reset_index(drop=True)
categories_unique['category_id'] = categories_unique.index + 1
# Rename the column to match the SQL schema expectation
categories_unique.rename(columns={'category': 'category_name'}, inplace=True)
categories_unique.to_sql('categories', conn, if_exists='append', index=False)

# Then, prepare books data with category_id
books_for_db = pd.merge(df, categories_unique, left_on='category', right_on='category_name', how='left')
books_for_db = books_for_db[['title', 'price_GBP', 'price_INR', 'rating', 'in_stock', 'category_id']]
books_for_db.rename(columns={'rating': 'rating'}, inplace=True)
books_for_db.to_sql('books', conn, if_exists='append', index=False)

print("Data inserted into SQLite tables.")

# 5. Let's execute the five required queries
# Query 1: SELECT and LIMIT - Get the first 5 books

query_1 = "SELECT title, price_GBP FROM books LIMIT 5;"
output_1 = pd.read_sql_query(query_1, conn)
print("Query 1 Output:\n", output_1)



# Query 2: WHERE and ORDER BY - Get books in stock priced over 10 GBP, ordered by price
query_2 = """
SELECT title, price_GBP FROM books
WHERE in_stock = 1 AND price_GBP > 10.00
ORDER BY price_GBP DESC;
"""
output_2 = pd.read_sql_query(query_2, conn)
print("Query 2 Output:\n", output_2)

# Query 3: DISTINCT and BETWEEN - Get unique ratings for books priced between 5 and 15 GBP
query_3 = """
SELECT DISTINCT rating FROM books
WHERE price_GBP BETWEEN 5.00 AND 15.00;
"""
output_3 = pd.read_sql_query(query_3, conn)
print("Query 3 Output:\n", output_3)


# Query 4: IN - Get books that fall into specific category IDs (e.g., 1, 3, or 5)
query_4 = "SELECT title, category_id FROM books WHERE category_id IN (1, 3, 5);"
output_4 = pd.read_sql_query(query_4, conn)
print("Query 4 Output:\n", output_4)



# Query 5: JOIN - List the 10 highest-rated books along with their category names
query_5 = """
SELECT b.title, c.category_name, b.rating
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC
LIMIT 10;
"""

output_5 = pd.read_sql_query(query_5, conn)
print("Query 5 Output:\n", output_5)

# Close the database connection.

# 6. Reproduce the join using pandas in-memory DataFrames
# Connect to the SQLite database (re-connecting to ensure context is clear)
conn = sqlite3.connect('books_database.db')
# 1. Read the join query back into a DataFrame
query_5_sql = """
SELECT b.title, c.category_name, b.rating
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC
LIMIT 10;
"""

df_from_sql = pd.read_sql_query(query_5_sql, conn)
# 2. Reproduce the join using pandas in-memory DataFrames
# Assuming 'books_df' and 'categories_df' are already in memory from earlier steps
# The current dataframes 'df' (final_books_data) and 'categories_unique' can be used

df_merged_pandas = pd.merge(books_for_db, categories_unique, on='category_id', how='left')
df_merged_pandas = df_merged_pandas[['title', 'category_name', 'rating']].sort_values(by='rating', ascending=False).head(10)

comparison = pd.concat(
    [
        df_from_sql.reset_index(drop=True),
        df_merged_pandas.reset_index(drop=True)
    ],
    axis=1,
    keys=["SQL", "Pandas"]
)

print("\nSQL JOIN vs Pandas MERGE:\n", comparison)

# It's good practice to close the connection when done, especially in scripts
conn.close()