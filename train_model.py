import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Generate simple synthetic dataset
np.random.seed(42)

data_size = 500

age = np.random.randint(30, 80, data_size)
cholesterol = np.random.randint(150, 300, data_size)
blood_pressure = np.random.randint(90, 180, data_size)
max_heart_rate = np.random.randint(60, 200, data_size)

# Simple rule to create labels
# Create a more realistic scoring system
risk_score = (
    (age > 50).astype(int) +
    (cholesterol > 220).astype(int) +
    (blood_pressure > 130).astype(int) +
    (max_heart_rate < 80).astype(int)
)

# Convert to binary risk
risk = (risk_score >= 2).astype(int)

df = pd.DataFrame({
    "age": age,
    "cholesterol": cholesterol,
    "blood_pressure": blood_pressure,
    "max_heart_rate": max_heart_rate,
    "risk": risk
})

X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

joblib.dump(model, "heart_disease_model.pkl")

print("Heart Disease model trained and saved!")