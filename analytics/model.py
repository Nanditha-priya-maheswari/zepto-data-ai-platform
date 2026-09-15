import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Preparing data for machine learning...")
df = pd.read_csv('analytics/titanic.csv')
df = df.drop(columns=['deck'])
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked'])


features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch']
X = df[features].copy()
y = df['survived']


X['sex'] = LabelEncoder().fit_transform(X['sex'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Random Forest Classifier...\n")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("--- Evaluation Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, predictions))
