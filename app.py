import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# ===============================
# Load Model
# ===============================

model = joblib.load("models/heart_failure_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")


# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================

st.markdown("""
<style>

.big-title {
    font-size:40px;
    font-weight:bold;
    color:#d62728;
}

.subtitle {
    font-size:18px;
    color:gray;
}

.result-box {
    padding:20px;
    border-radius:10px;
    background-color:#f5f5f5;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# Header
# ===============================

st.markdown('<p class="big-title">❤️ Heart Failure Prediction</p>', unsafe_allow_html=True)

st.markdown(
"""
Aplikasi ini menggunakan **Machine Learning (Extra Trees Classifier)** untuk memprediksi
risiko **penyakit jantung** berdasarkan data kesehatan pasien.

Silakan isi data pasien pada form berikut lalu klik **Predict**.
"""
)


# ===============================
# Sidebar
# ===============================

st.sidebar.title("About Project")

st.sidebar.info(
"""
Model : Extra Trees Classifier  

Dataset : Heart Disease Dataset  

Metode :
- Data Preprocessing
- SMOTE for imbalance handling
- Ensemble Learning

Developer :
Syahid Khomsyi
"""
)


# ===============================
# Input Form
# ===============================

st.subheader("Patient Information")

col1, col2 = st.columns(2)


with col1:

    age = st.slider("Age", 20, 80, 40)

    sex_display = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    # Mapping ke format dataset
    sex = "M" if sex_display == "Male" else "F"

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_bp = st.slider(
        "Resting Blood Pressure",
        80, 200, 120
    )

    cholesterol = st.slider(
        "Cholesterol",
        100, 600, 200
    )


with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Max Heart Rate",
        60, 220, 150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["N", "Y"]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0, 6.0, 1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


# ===============================
# Encode Input
# ===============================

def encode_input(data):

    for col, encoder in encoders.items():
        data[col] = encoder.transform(data[col])

    return data

def get_risk_category(prob):

    if prob < 0.30:
        return "Low Risk", "green"
    elif prob < 0.60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def plot_risk_gauge(probability):

    percent = probability * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Heart Disease Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},
                {'range': [30, 60], 'color': "#f1c40f"},
                {'range': [60, 100], 'color': "#e74c3c"}
            ],
        }
    ))

    fig.update_layout(height=350)

    return fig

# ===============================
# Prediction Button
# ===============================

st.write("")

if st.button("Predict Heart Disease Risk"):

    input_data = pd.DataFrame({
        'Age':[age],
        'Sex':[sex],
        'ChestPainType':[chest_pain],
        'RestingBP':[resting_bp],
        'Cholesterol':[cholesterol],
        'FastingBS':[fasting_bs],
        'RestingECG':[resting_ecg],
        'MaxHR':[max_hr],
        'ExerciseAngina':[exercise_angina],
        'Oldpeak':[oldpeak],
        'ST_Slope':[st_slope]
    })

    # Encoding
    input_data = encode_input(input_data)

    # Scaling
    scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    risk_percent = probability * 100

    risk_label, color = get_risk_category(probability)

    st.write("")
    st.subheader("Prediction Result")

    col1, col2 = st.columns([1,1])

    with col1:

        if risk_label == "Low Risk":
            st.success(f"✅ {risk_label}")

        elif risk_label == "Moderate Risk":
            st.warning(f"⚠️ {risk_label}")

        else:
            st.error(f"🚨 {risk_label}")

        st.write(f"### Estimated Risk: **{risk_percent:.1f}%**")

        st.info(
            f"""
Interpretation:

Based on the patient's health indicators, the model estimates a **{risk_percent:.1f}% probability**
of heart disease.

Risk Category: **{risk_label}**
"""
        )

    with col2:

        gauge = plot_risk_gauge(probability)
        st.plotly_chart(gauge, use_container_width=True)

# ===============================
# Footer
# ===============================

st.write("---")

st.caption("Machine Learning Heart Failure Prediction | Streamlit Deployment")
