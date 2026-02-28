import gradio as gr
import joblib
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import os
import math

# Load the model
try:
    model = joblib.load("heart_disease_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit if model missing
    model = None

# ---------------------------
# Prediction & Analysis Logic
# ---------------------------

def calculate_heart_rate_score(max_hr, age):
    """
    Calculates a simple heart rate efficacy score based on age-predicted max heart rate (220 - age).
    Returns score (0-100) and interpretation.
    """
    predicted_max = 220 - age
    ratio = max_hr / predicted_max if predicted_max > 0 else 0
    
    score = min(100, max(0, int(ratio * 100)))
    
    if 0.85 <= ratio <= 1.05:
        return score, "Excellent", "#22c55e"
    elif 0.70 <= ratio < 0.85:
        return score, "Good", "#3b82f6"
    else:
        return score, "Concerning", "#ef4444"

def analyze(age, cholesterol, blood_pressure, max_heart_rate):
    if model is None:
        return 0, "MODEL ERROR", "#64748b", "Please ensure model is trained."

    input_data = pd.DataFrame(
        [[age, cholesterol, blood_pressure, max_heart_rate]],
        columns=["age", "cholesterol", "blood_pressure", "max_heart_rate"]
    )

    probability = model.predict_proba(input_data)[0][1] * 100
    hr_score, hr_label, hr_color = calculate_heart_rate_score(max_heart_rate, age)

    if probability < 30:
        risk = "LOW RISK"
        color = "#22c55e" # Green
        advice = "Your cardiovascular profile looks strong. Keep up the healthy habits!"
    elif probability < 60:
        risk = "MODERATE RISK"
        color = "#f59e0b" # Amber
        advice = "Some elevated indicators. Consider increasing cardiovascular exercise and reducing sodium."
    else:
        risk = "HIGH RISK"
        color = "#ef4444" # Red
        advice = "A medical consultation is strongly advised. Focus on heart-healthy diet and stress management."

    return probability, risk, color, advice, hr_score, hr_label, hr_color


# ---------------------------
# Component Generators
# ---------------------------

def get_gauge_html(probability, color):
    # Calculate SVG path for semi-circle gauge
    # Using 180 degrees range
    angle = (probability / 100) * 180
    # Center 100,100; Radius 80
    # Start at 0, 100 (left)
    # End at 200, 100 (right)
    # We want it to build clockwise
    return f"""
    <div class="gauge-container">
        <svg viewBox="0 0 200 120" class="gauge">
            <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#1e293b" stroke-width="12" stroke-linecap="round"/>
            <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="{color}" stroke-width="12" 
                stroke-dasharray="251.32" stroke-dashoffset="{251.32 * (1 - probability/100)}" 
                stroke-linecap="round" class="gauge-fill"/>
            <text x="100" y="85" text-anchor="middle" class="gauge-percentage" fill="white">{probability:.1f}%</text>
            <text x="100" y="110" text-anchor="middle" class="gauge-label" fill="#94a3b8">Probability</text>
        </svg>
    </div>
    """

def get_pulse_html(max_hr, hr_color):
    # Animation speed calculation (Max HR/60 bpm = seconds per beat)
    # Lower seconds = faster animation
    duration = 60 / max_hr if max_hr > 0 else 1
    return f"""
    <div class="pulse-card">
        <div class="pulse-icon" style="animation-duration: {duration:.2f}s; color: {hr_color};">❤️</div>
        <div class="pulse-value">{max_hr} <span class="unit">BPM</span></div>
        <div class="pulse-label">Maximum Heart Rate</div>
    </div>
    """

def predict_ui(age, cholesterol, blood_pressure, max_heart_rate):
    prob, risk, color, advice, hr_score, hr_label, hr_color = analyze(age, cholesterol, blood_pressure, max_heart_rate)
    
    gauge_html = get_gauge_html(prob, color)
    pulse_html = get_pulse_html(max_heart_rate, hr_color)
    
    return f"""
    <div class="dashboard">
        <div class="main-stats">
            <div class="risk-card" style="border-top: 4px solid {color};">
                <div class="risk-title" style="color: {color};">{risk}</div>
                {gauge_html}
                <div class="advice-text">{advice}</div>
            </div>
            
            <div class="vitals-grid">
                {pulse_html}
                <div class="stat-card">
                    <div class="stat-value" style="color: {hr_color};">{hr_score}%</div>
                    <div class="stat-label">HR Efficiency</div>
                    <div class="stat-sub">{hr_label}</div>
                </div>
            </div>
        </div>
        
        <div class="metrics-summary">
            <div class="metric">
                <span class="m-label">Age</span>
                <span class="m-value">{age}y</span>
            </div>
            <div class="metric">
                <span class="m-label">Cholesterol</span>
                <span class="m-value">{cholesterol} mg/dL</span>
            </div>
            <div class="metric">
                <span class="m-label">Blood Pressure</span>
                <span class="m-value">{blood_pressure} mmHg</span>
            </div>
        </div>
    </div>
    """

# ---------------------------
# PDF REPORT GENERATOR
# ---------------------------

def generate_report(age, cholesterol, blood_pressure, max_heart_rate):
    prob, risk, color, advice, hr_score, hr_label, _ = analyze(age, cholesterol, blood_pressure, max_heart_rate)
    file_path = "heart_health_report.pdf"
    
    doc = SimpleDocTemplate(file_path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor=colors.HexColor("#0f172a"))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=16, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor("#334155"))
    body_style = styles['Normal']
    
    # Header
    elements.append(Paragraph("Heart Health Intelligence Report", title_style))
    elements.append(Paragraph(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Patient Vitals Table
    elements.append(Paragraph("Input Vitals", h2_style))
    data = [
        ["Metric", "Value"],
        ["Age", f"{age} years"],
        ["Cholesterol", f"{cholesterol} mg/dL"],
        ["Systolic Blood Pressure", f"{blood_pressure} mmHg"],
        ["Maximum Heart Rate", f"{max_heart_rate} BPM"]
    ]
    t = Table(data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#64748b")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Risk Results
    elements.append(Paragraph("AI Assessment Results", h2_style))
    risk_color = colors.HexColor(color)
    elements.append(Paragraph(f"<b>Overall Risk: {risk}</b>", ParagraphStyle('Risk', parent=body_style, textColor=risk_color, fontSize=14)))
    elements.append(Paragraph(f"Calculated Probability: {prob:.1f}%", body_style))
    elements.append(Paragraph(f"Heart Rate Efficiency: {hr_score}% ({hr_label})", body_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    elements.append(Paragraph("Clinical Advice:", ParagraphStyle('AdviceTitle', parent=body_style, fontName='Helvetica-Bold')))
    elements.append(Paragraph(advice, body_style))
    
    # Disclaimer
    elements.append(Spacer(1, 0.5 * inch))
    disclaimer = "<i>Disclaimer: This report is generated by an AI model and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider.</i>"
    elements.append(Paragraph(disclaimer, ParagraphStyle('Disclaimer', parent=body_style, fontSize=8, textColor=colors.grey)))
    
    doc.build(elements)
    return file_path


# ---------------------------
# UI Layout
# ---------------------------

with gr.Blocks(
    title="HeartAI Intelligence",
) as demo:

    with gr.Column(elem_classes="container"):
        
        with gr.Column(elem_classes="header-section"):
            gr.HTML("""
                <h1 class="header-title">HeartAI Intelligence</h1>
                <p class="header-subtitle">Real-time Advanced Cardiovascular Risk Assessment</p>
            """)

        with gr.Tabs():
            
            with gr.Tab("🔍 Diagnostic Dashboard"):
                
                with gr.Row():
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛠 Assessment Parameters")
                        age = gr.Slider(20, 90, value=45, step=1, label="Age", info="Years")
                        cholesterol = gr.Slider(100, 400, value=200, step=1, label="Cholesterol", info="mg/dL")
                        blood_pressure = gr.Slider(80, 220, value=120, step=1, label="Systolic BP", info="mmHg")
                        max_heart_rate = gr.Slider(50, 220, value=150, step=1, label="Max Heart Rate", info="BPM (Achieved under stress)")
                        
                        gr.Markdown("""
                        > [!TIP]
                        > These parameters are used by our Decision Tree model to estimate potential heart disease probability.
                        """)

                    with gr.Column(scale=2):
                        # The dynamic output area
                        output = gr.HTML(label="Results")

                # State update triggers
                inputs = [age, cholesterol, blood_pressure, max_heart_rate]
                
                # Update on change
                for input_comp in inputs:
                    input_comp.change(predict_ui, inputs, output)
                
                # Run once on load
                demo.load(predict_ui, inputs, output)

            with gr.Tab("📄 Clinical Report"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### Generate Patient Summary
                        Finalize the assessment and download a professional PDF report containing the inputs and AI interpretation.
                        """)
                        gen_btn = gr.Button("Generate Professional PDF", variant="primary", size="lg")
                    
                    with gr.Column():
                        file_out = gr.File(label="Download Report")

                gen_btn.click(generate_report, inputs, file_out)

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;900&display=swap');

:root {
    --bg-dark: #020617;
    --card-bg: rgba(15, 23, 42, 0.6);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-blue: #3b82f6;
}

body {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 100%);
    font-family: 'Outfit', sans-serif;
    color: white;
    margin: 0;
}

.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 20px;
}

.header-section {
    text-align: center;
    margin-bottom: 50px;
}

.header-title {
    font-size: 56px;
    font-weight: 900;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.header-subtitle {
    color: #64748b;
    font-size: 20px;
    margin-top: 10px;
}

/* Dashboard Layout */
.dashboard {
    display: flex;
    flex-direction: column;
    gap: 25px;
    animation: fadeIn 0.8s ease-out;
}

.main-stats {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 25px;
}

.risk-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 28px;
    padding: 35px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

.risk-title {
    font-size: 32px;
    font-weight: 900;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.gauge-container {
    width: 280px;
    height: 180px;
}

.gauge-fill {
    transition: stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.gauge-percentage {
    font-size: 36px;
    font-weight: 800;
}

.gauge-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.advice-text {
    margin-top: 20px;
    font-size: 16px;
    color: #cbd5e1;
    max-width: 80%;
    line-height: 1.6;
}

.vitals-grid {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.pulse-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 28px;
    padding: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.pulse-icon {
    font-size: 48px;
    animation: heartBeat infinite ease-in-out;
}

.pulse-value {
    font-size: 42px;
    font-weight: 900;
    margin-top: 10px;
}

.pulse-value .unit {
    font-size: 16px;
    color: #64748b;
    font-weight: 400;
}

.stat-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 28px;
    padding: 30px;
    text-align: center;
}

.stat-value {
    font-size: 48px;
    font-weight: 900;
}

.stat-label {
    color: #94a3b8;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 5px;
}

.stat-sub {
    font-size: 18px;
    font-weight: 600;
    margin-top: 5px;
}

.metrics-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 20px;
}

.metric {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.m-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
}

.m-value {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
}

@keyframes heartBeat {
    0% { transform: scale(1); }
    15% { transform: scale(1.3); }
    30% { transform: scale(1); }
    45% { transform: scale(1.15); }
    60% { transform: scale(1); }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Override Gradio Slider Styles */
.gr-slider input {
    accent-color: var(--accent-blue);
}

.gradio-container {
    border: none !important;
}

.tabs { border: none !important; }
.tabitem { background: transparent !important; }
"""

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", show_error=True, css=GLOBAL_CSS)
