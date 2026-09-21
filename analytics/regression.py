import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Load & Clean
df = sns.load_dataset('titanic')
df.drop(columns=['deck', 'alive', 'embark_town', 'class', 'who', 'adult_male'], inplace=True, errors='ignore')
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.dropna(inplace=True)

# Target is Fare, removing Survived to prevent data leakage in regression
X = df.drop(columns=['fare', 'survived']) 
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'pclass']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create and train regression pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("--- Multivariate Linear Regression Metrics (Predicting Fare) ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-Squared: {r2:.2f}")
print(f"Adjusted R-Squared: {adj_r2:.2f}")

# Generate Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot (Predicted Fare vs Error)')
plt.xlabel('Predicted Fare')
plt.ylabel('Residuals')
plt.savefig('analytics/residual_plot.png')
print("--- Success: Residual plot saved to analytics/residual_plot.png ---")