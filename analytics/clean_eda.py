import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("Loading offline dataset...")
df = pd.read_csv('analytics/titanic.csv')


print("\n--- Starting Data Cleaning ---")

df = df.drop(columns=['deck'])


median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)


df = df.dropna(subset=['embarked'])


df = df.drop_duplicates()

print("Cleaning Complete. New Dataset Shape:", df.shape)


print("\nGenerating EDA Charts...")

sns.set_theme(style="whitegrid")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


sns.countplot(data=df, x='survived', ax=axes[0], palette='Set2')
axes[0].set_title('Overall Survival Count')


sns.countplot(data=df, x='survived', hue='sex', ax=axes[1], palette='pastel')
axes[1].set_title('Survival by Sex')


sns.histplot(data=df, x='age', kde=True, ax=axes[2], color='skyblue', bins=30)
axes[2].set_title('Age Distribution')


plt.tight_layout()
plt.savefig('analytics/eda_charts.png')
print("Saved charts to analytics/eda_charts.png")


plt.show()
