import joblib
import pandas as pd
import os

print("Testing joblib load...")
try:
    model = joblib.load("heart_disease_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

print("Testing pandas dataframe...")
df = pd.DataFrame([[45, 200, 120, 150]], columns=["age", "cholesterol", "blood_pressure", "max_heart_rate"])
print("Dataframe created.")

if 'model' in locals():
    print("Testing prediction...")
    try:
        prob = model.predict_proba(df)[0][1]
        print(f"Prediction success: {prob}")
    except Exception as e:
        print(f"Prediction failed: {e}")

from reportlab.platypus import SimpleDocTemplate
print("Reportlab import success.")

import gradio as gr
print("Gradio import success.")
