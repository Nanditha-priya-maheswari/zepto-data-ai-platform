import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

# 1. Load & Clean
df = sns.load_dataset('titanic')
df.drop(columns=['deck', 'alive', 'embark_town', 'class', 'who', 'adult_male'], inplace=True, errors='ignore')
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.dropna(inplace=True)

# 2. EDA (4 Required Charts)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df['age'], kde=True, ax=axes[0,0]).set_title('Age Distribution (Histogram)')
sns.boxplot(x='pclass', y='fare', data=df, ax=axes[0,1]).set_title('Fare by Class (Box Plot)')
sns.countplot(x='survived', hue='sex', data=df, ax=axes[1,0]).set_title('Survival by Sex (Count Plot)')
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=axes[1,1]).set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('analytics/eda_charts.png')
print("--- EDA Complete: Saved to analytics/eda_charts.png ---\n")

# 3. Setup Preprocessing Pipeline (No Data Leakage)
X = df.drop(columns=['survived'])
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'pclass']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 4. Three Models Evaluation (Baseline)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

print("--- Baseline Model Comparison ---")
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    print(f"{name}: Acc={acc:.2f} | F1={f1:.2f} | ROC-AUC={auc:.2f}")

# 5. Class Imbalance (SMOTE) & GridSearchCV on Random Forest
print("\n--- Tuning Random Forest with SMOTE ---")
# Using imblearn Pipeline to apply SMOTE only to training data during cross-validation
rf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, oob_score=True))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 5, 10],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"OOB Score from best model: {best_model.named_steps['classifier'].oob_score_:.2f}")
print(f"\n--- Final Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_best):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_best):.2f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_best):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_best):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")

# 6. Pipeline Persistence
joblib.dump(best_model, 'analytics/best_rf_pipeline.pkl')
print("\n--- Success: Complete pipeline (Preprocessing + SMOTE + Model) saved to analytics/best_rf_pipeline.pkl ---")
