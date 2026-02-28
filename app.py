import gradio as gr
import joblib
import pandas as pd

model = joblib.load("heart_disease_model.pkl")


def analyze(age, cholesterol, blood_pressure, max_heart_rate):

    input_data = pd.DataFrame(
        [[age, cholesterol, blood_pressure, max_heart_rate]],
        columns=["age", "cholesterol", "blood_pressure", "max_heart_rate"]
    )

    probability = model.predict_proba(input_data)[0][1] * 100

    if probability < 30:
        risk = "LOW RISK"
        color = "#22c55e"
        interpretation = "Healthy range. Maintain lifestyle."
    elif probability < 60:
        risk = "MODERATE RISK"
        color = "#f59e0b"
        interpretation = "Monitor diet, exercise regularly."
    else:
        risk = "HIGH RISK"
        color = "#ef4444"
        interpretation = "Consult a cardiologist immediately."

    return probability, risk, color, interpretation


def predict(age, cholesterol, blood_pressure, max_heart_rate):

    probability, risk, color, interpretation = analyze(
        age, cholesterol, blood_pressure, max_heart_rate
    )

    return f"""
    <div class="result-card">

        <div class="risk-label" style="color:{color};">
            {risk}
        </div>

        <div class="percentage" id="counter">
            {probability:.1f}%
        </div>

        <div class="bar-container">
            <div class="bar-fill" style="width:{probability}%;"></div>
        </div>

        <div class="legend">
            🟢 0–29% &nbsp; 🟡 30–59% &nbsp; 🔴 60%+
        </div>

        <div class="interpretation">
            {interpretation}
        </div>

        <div class="pulse">❤️</div>

    </div>
    """


with gr.Blocks(css="""
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    font-family: 'Segoe UI', sans-serif;
}

.result-card {
    padding: 40px;
    border-radius: 25px;
    background: linear-gradient(145deg,#111827,#1f2937);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    text-align:center;
    transition: transform 0.3s ease;
}
.result-card:hover {
    transform: scale(1.02);
}

.risk-label {
    font-size: 42px;
    font-weight: 900;
}

.percentage {
    font-size: 55px;
    font-weight: 900;
    margin-top: 15px;
}

.bar-container {
    height: 18px;
    width: 100%;
    background: #0f172a;
    border-radius: 50px;
    overflow: hidden;
    margin-top: 25px;
}

.bar-fill {
    height: 100%;
    background: linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);
    transition: width 1s ease-in-out;
}

.legend {
    margin-top: 12px;
    font-size: 14px;
    color: #cbd5e1;
}

.interpretation {
    margin-top: 20px;
    font-size: 18px;
    color: #e2e8f0;
}

.pulse {
    margin-top: 25px;
    font-size: 30px;
    animation: pulse 1.2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.3); }
    100% { transform: scale(1); }
}
""") as demo:

    gr.Markdown("""
    <h1 style='
        text-align:center;
        font-size:55px;
        font-weight:900;
        background:linear-gradient(90deg,#22c55e,#3b82f6,#ef4444);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
    '>
        ❤️ AI Heart Health Intelligence
    </h1>
    <p style='text-align:center;color:#94a3b8;font-size:18px;'>
        Real-time Cardiovascular Risk Assessment
    </p>
    """)

    with gr.Row():

        with gr.Column(scale=1):
            age = gr.Slider(20, 90, step=1, label="Age")
            cholesterol = gr.Slider(100, 350, step=1, label="Cholesterol")
            blood_pressure = gr.Slider(80, 200, step=1, label="Blood Pressure")
            max_heart_rate = gr.Slider(50, 200, step=1, label="Max Heart Rate")

        with gr.Column(scale=1):
            output = gr.HTML("""
            <div style="
                padding:40px;
                border-radius:25px;
                background:rgba(255,255,255,0.03);
                text-align:center;
                color:#64748b;
                font-size:18px;
            ">
                Adjust sliders to see live AI prediction
            </div>
            """)

    age.change(predict, [age, cholesterol, blood_pressure, max_heart_rate], output)
    cholesterol.change(predict, [age, cholesterol, blood_pressure, max_heart_rate], output)
    blood_pressure.change(predict, [age, cholesterol, blood_pressure, max_heart_rate], output)
    max_heart_rate.change(predict, [age, cholesterol, blood_pressure, max_heart_rate], output)


demo.launch()