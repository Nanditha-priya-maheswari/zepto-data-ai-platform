import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3


# 1. Scrape 60 Books across 3 Categories
categories = {
    'Travel': 'https://books.toscrape.com/catalogue/category/books/travel_2/index.html',
    'Mystery': 'https://books.toscrape.com/catalogue/category/books/mystery_3/index.html',
    'Historical Fiction': 'https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html'
}

books_data = []
for cat_name, url in categories.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Each category page has 20 books (3 categories * 20 = 60 books)
    for article in soup.find_all('article', class_='product_pod'):
        title = article.h3.a['title']
        price_text = article.find('p', class_='price_color').text
        price_gbp = float(price_text.replace('£', '').replace('Â', ''))
        rating = article.p['class'][1]
        availability = article.find('p', class_='instock availability').text.strip()
        
        books_data.append({
            'title': title,
            'price_gbp': price_gbp,
            'star_rating': rating,
            'availability': availability,
            'category': cat_name
        })

df_books = pd.DataFrame(books_data)

# 2. Data Cleaning & Business Logic
# Convert GBP to INR (Assuming 1 GBP = 105.50 INR)
df_books['price_inr'] = df_books['price_gbp'] * 105.50 
df_books['in_stock'] = df_books['availability'].apply(lambda x: 1 if 'In stock' in x else 0)
df_books.drop(columns=['availability'], inplace=True)

# Map star ratings to integers
rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
df_books['star_rating'] = df_books['star_rating'].map(rating_map)

# Create Categories DataFrame for Relational DB
df_categories = pd.DataFrame({'category_name': list(categories.keys())})
df_categories['category_id'] = range(1, len(df_categories) + 1)

# Merge category_id back to books and drop text category
df_books = df_books.merge(df_categories, left_on='category', right_on='category_name')
df_books.drop(columns=['category', 'category_name'], inplace=True)
df_books['book_id'] = range(1, len(df_books) + 1)

# 3. SQLite Database Integration (PK/FK)
conn = sqlite3.connect('data_pipeline/zepto_catalog.db')

cursor = conn.cursor()

# Drop existing tables to ensure clean schema creation
cursor.execute('DROP TABLE IF EXISTS Books')
cursor.execute('DROP TABLE IF EXISTS Categories')

# Create strictly typed tables with Primary and Foreign Keys
cursor.execute('''
CREATE TABLE Categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE Books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    star_rating INTEGER,
    price_inr REAL,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES Categories (category_id)
)
''')

# Insert Data (using append to respect the schema created above)
df_categories.to_sql('Categories', conn, if_exists='append', index=False)
df_books.to_sql('Books', conn, if_exists='append', index=False)

# 4. Required SQL Queries
queries = {
    "1. SELECT/WHERE & ORDER BY": "SELECT title, price_gbp FROM Books WHERE star_rating = 5 ORDER BY price_gbp DESC LIMIT 5;",
    "2. LIMIT": "SELECT * FROM Books LIMIT 3;",
    "3. DISTINCT": "SELECT DISTINCT star_rating FROM Books;",
    "4. IN/BETWEEN": "SELECT title, price_gbp FROM Books WHERE price_gbp BETWEEN 20 AND 40;",
    "5. JOIN": """
        SELECT b.title, b.price_gbp, c.category_name 
        FROM Books b
        JOIN Categories c ON b.category_id = c.category_id
        LIMIT 5;
    """
}

print("--- SQL Query Results ---")
for desc, query in queries.items():
    print(f"\n{desc}:")
    # Read SQL directly into Pandas
    print(pd.read_sql(query, conn))

# 5. Pandas Merge (Reproducing the JOIN)
print("\n--- Pandas Merge Result ---")
merged_df = pd.merge(df_books, df_categories, on='category_id', how='inner')
print(merged_df[['title', 'price_gbp', 'category_name']].head(5))

conn.close()
print("\n--- Success: Pipeline completed and zepto_catalog.db created ---")