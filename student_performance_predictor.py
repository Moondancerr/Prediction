# ------------------------------
# Student Performance Predictor
# ------------------------------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# 1. Load Dataset
# ------------------------------
data = pd.read_csv('StudentsPerformance.csv')
print("✅ Dataset loaded successfully!")

# ------------------------------
# 2. Exploratory Data Analysis
# ------------------------------
print("\n📊 Dataset Preview:")
print(data.head())

print("\n📄 Data Description:")
print(data.describe(include='all'))

print("\nℹ️ Dataset Info:")
print(data.info())

# Optional: Visualize score distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['math score'], kde=True, bins=20)
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ------------------------------
# 3. Preprocessing
# ------------------------------
features = ['reading score', 'writing score', 
            'gender', 'race/ethnicity', 
            'parental level of education', 'lunch', 
            'test preparation course']
target = 'math score'

df = data[features + [target]].copy()

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=[
    'gender', 
    'race/ethnicity',
    'parental level of education', 
    'lunch', 
    'test preparation course'
], drop_first=True)

print("\n🧾 Encoded Features:")
print(df_encoded.columns.tolist())

# ------------------------------
# 4. Train/Test Split
# ------------------------------
X = df_encoded.drop('math score', axis=1)
y = df_encoded['math score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📦 Train size: {X_train.shape}, Test size: {X_test.shape}")

# ------------------------------
# 5. Train the Model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# ------------------------------
# 6. Evaluate the Model
# ------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ------------------------------
# 7. Visualization
# ------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted Math Scores")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

# ------------------------------
# 8. Feature Importance
# ------------------------------
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n🧠 Feature Importance:")
print(coef_df)
