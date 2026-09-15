import os
import seaborn as sns
import pandas as pd

# Ensure the analytics folder exists
os.makedirs('analytics', exist_ok=True)

# 1. Load dataset and save local fallback (Required by rubric)
print("Fetching Titanic dataset...")
df = sns.load_dataset('titanic')
df.to_csv('analytics/titanic.csv', index=False)
print("Saved offline fallback to analytics/titanic.csv\n")

# 2. Initial Data Profiling
print("--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

print("--- Dataset Info ---")
print(df.info())
print("\n")

print("--- Missing Values Percentage ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_display = missing_pct[missing_pct > 0].sort_values(ascending=False)
print(missing_display.round(2).astype(str) + '%\n')

print("--- First 5 Rows ---")
print(df.head())