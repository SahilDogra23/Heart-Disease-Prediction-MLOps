import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")
st.title("🫀 Heart Disease Predictor")
st.markdown("Enter patient details below to predict the likelihood of heart disease.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=52)
    trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=125)
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T abnormality", "LV hypertrophy"][x])
    slope = st.selectbox("ST Slope", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

with col2:
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=212)
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=168)
    ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])

with col3:
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"][x])
    fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect", "Unknown"][x])

oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
st.divider()

if st.button("Predict", use_container_width=True, type="primary"):
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        if result["prediction"] == 1:
            st.error(f"**{result['result']}**")
        else:
            st.success(f"**{result['result']}**")
        st.metric(label="Model Confidence", value=f"{result['confidence']}%")
        st.caption("This tool is for educational purposes only. Always consult a medical professional.")
    except Exception as e:
        st.error(f"Could not connect to API. Make sure uvicorn is running! {e}")