import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the offline dataset you saved earlier
print("Loading offline dataset...")
df = pd.read_csv('analytics/titanic.csv')

# --- DATA CLEANING ---
print("\n--- Starting Data Cleaning ---")
# Drop 'deck' because it has >70% missing values
df = df.drop(columns=['deck'])

# Fill missing 'age' values with the median age
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)

# Drop the 2 rows where 'embarked' is missing
df = df.dropna(subset=['embarked'])

# Drop duplicate rows if any exist
df = df.drop_duplicates()

print("Cleaning Complete. New Dataset Shape:", df.shape)

# --- EXPLORATORY DATA ANALYSIS (EDA) ---
print("\nGenerating EDA Charts...")
# Set visual style
sns.set_theme(style="whitegrid")

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Overall Survival (0 = No, 1 = Yes)
sns.countplot(data=df, x='survived', ax=axes[0], palette='Set2')
axes[0].set_title('Overall Survival Count')

# Chart 2: Survival by Sex
sns.countplot(data=df, x='survived', hue='sex', ax=axes[1], palette='pastel')
axes[1].set_title('Survival by Sex')

# Chart 3: Age Distribution
sns.histplot(data=df, x='age', kde=True, ax=axes[2], color='skyblue', bins=30)
axes[2].set_title('Age Distribution')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('analytics/eda_charts.png')
print("Saved charts to analytics/eda_charts.png")

# Show the plot on screen
plt.show()