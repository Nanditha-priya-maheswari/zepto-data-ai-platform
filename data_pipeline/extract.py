import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://books.toscrape.com/"
CATEGORIES = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "catalogue/category/books/historical-fiction_4/index.html",
}

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
GBP_TO_INR = 105.50  # Required fixed rate

all_books = []
print("Starting extraction...")

for cat_name, rel_url in CATEGORIES.items():
    url = BASE_URL + rel_url
    
    # Scrape pages until we have at least 25 books per category
    while url and len([b for b in all_books if b['category'] == cat_name]) < 25:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for article in soup.select("article.product_pod"):
            # Extract raw text
            title = article.h3.a["title"]
            price_raw = article.select_one(".price_color").get_text()
            rating_raw = [c for c in article.select_one(".star-rating")["class"] if c != "star-rating"][0]
            avail_raw = article.select_one(".availability").get_text().strip()
            
            # Clean and convert
            price_gbp = float(re.sub(r"[^\d.]", "", price_raw))
            
            all_books.append({
                "title": title,
                "price_gbp": price_gbp,
                "price_inr": round(price_gbp * GBP_TO_INR, 2),
                "rating": RATING_MAP.get(rating_raw, 0),
                "in_stock": int("in stock" in avail_raw.lower()),
                "category": cat_name
            })
            
        # Move to the next page if it exists
        next_btn = soup.select_one("li.next a")
        url = "/".join(url.split("/")[:-1]) + "/" + next_btn["href"] if next_btn else None

df = pd.DataFrame(all_books)
print(f"\nExtraction complete! Total books scraped: {len(df)}")
print(df.head())


import sqlite3

print("\n--- Starting Database Loading ---")
# 1. Connect to SQLite and Create Tables
conn = sqlite3.connect('zepto_catalog.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS categories (
                    category_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    category_name TEXT UNIQUE)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS books (
                    book_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    title TEXT, price_gbp REAL, price_inr REAL, 
                    rating INTEGER, in_stock INTEGER, 
                    category_id INTEGER REFERENCES categories(category_id))''')

# 2. Insert Categories and merge IDs back to the main DataFrame
cursor.executemany('INSERT OR IGNORE INTO categories (category_name) VALUES (?)', [(c,) for c in df['category'].unique()])
cat_df = pd.read_sql("SELECT * FROM categories", conn)
df = df.merge(cat_df, left_on='category', right_on='category_name')

# 3. Insert Books
books_data = df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']].values.tolist()
cursor.executemany('''INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) 
                      VALUES (?, ?, ?, ?, ?, ?)''', books_data)
conn.commit()
print("Data successfully loaded into zepto_catalog.db!\n")

# 4. Execute the 5 Required SQL Queries
queries = {
    "1. SELECT/WHERE": "SELECT title, price_gbp FROM books WHERE rating = 5 LIMIT 3;",
    "2. ORDER BY/LIMIT": "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 3;",
    "3. DISTINCT": "SELECT DISTINCT rating FROM books;",
    "4. BETWEEN": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20 LIMIT 3;",
    "5. JOIN": "SELECT c.category_name, b.title, b.rating FROM books b JOIN categories c ON b.category_id = c.category_id WHERE b.rating = 5 LIMIT 5;"
}

print("--- Running Required SQL Queries ---")
for name, query in queries.items():
    print(f"\n{name}:\n{pd.read_sql(query, conn)}")

# 5. Validate with Pandas Merge (Rubric requirement)
print("\n--- Validating SQL JOIN with Pandas Merge ---")
sql_join_result = pd.read_sql(queries["5. JOIN"], conn)

# Reproducing the exact same join using Pandas DataFrames
books_db = pd.read_sql("SELECT * FROM books", conn)
cats_db = pd.read_sql("SELECT * FROM categories", conn)
pandas_merge = pd.merge(books_db, cats_db, on='category_id')
pandas_result = pandas_merge[pandas_merge['rating'] == 5][['category_name', 'title', 'rating']].head(5)

# Check if SQL output matches Pandas output
print("SQL Output Matches Pandas Output:", sql_join_result.equals(pandas_result.reset_index(drop=True)))
conn.close()