
import streamlit as st
import pandas as pd
import joblib
import requests

## Loading the trained Model

model = joblib.load("Logistic_Regression.pkl")  

## Groq API set up 

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_treatment_recommendation(patient_data: dict, risk_level: str) -> str:
    """Send patient data and risk to Groq LLM for treatment recommendation."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    The following patient data is provided:
    {patient_data}

    The model predicts the lung cancer risk as: {risk_level}.

    Based on standard clinical knowledge, suggest treatment guidance or next steps
    suitable for a patient with this risk level. Keep the response clear and empathetic.
    """

    data = {
        "model": "llama3-8b-8192",  
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant (not a doctor)."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 500
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error from Groq API: {response.text}"



##STREAMLIT UI

st.set_page_config(page_title=" Lung Cancer Risk & Treatment", layout="centered")
st.title(" Lung Cancer Risk Predictor with AI Treatment Guidance")

with st.form("patient_form"):
    age = st.slider("Age", 1, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    air_pollution = st.slider("Air Pollution", 0, 10, 5)
    alcohol_use = st.slider("Alcohol use", 0, 10, 5)
    dust_allergy = st.slider("Dust Allergy", 0, 10, 5)
    occupational_hazards = st.slider("Occupational Hazards", 0, 10, 5)
    genetic_risk = st.slider("Genetic Risk", 0, 10, 5)
    chronic_lung_disease = st.slider("Chronic Lung Disease", 0, 10, 5)
    balanced_diet = st.slider("Balanced Diet", 0, 10, 5)
    obesity = st.slider("Obesity", 0, 10, 5)
    smoking = st.slider("Smoking", 0, 10, 5)
    passive_smoker = st.slider("Passive Smoker", 0, 10, 5)  # ✅ fixed
    chest_pain = st.slider("Chest Pain", 0, 10, 5)
    coughing_of_blood = st.slider("Coughing of Blood", 0, 10, 5)
    fatigue = st.slider("Fatigue", 0, 10, 5)
    weight_loss = st.slider("Weight Loss", 0, 10, 5)
    shortness_of_breath = st.slider("Shortness of Breath", 0, 10, 5)
    wheezing = st.slider("Wheezing", 0, 10, 5)
    swallowing_difficulty = st.slider("Swallowing Difficulty", 0, 10, 5)
    clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails", 0, 10, 5)
    frequent_cold = st.slider("Frequent Cold", 0, 10, 5)
    dry_cough = st.slider("Dry Cough", 0, 10, 5)
    snoring = st.slider("Snoring", 0, 10, 5)

    submitted = st.form_submit_button("Predict & Get Treatment")

if submitted:
    gender_val = 1 if gender == "Male" else 2

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_val],
        "Air Pollution": [air_pollution],
        "Alcohol use": [alcohol_use],
        "Dust Allergy": [dust_allergy],
        "OccuPational Hazards": [occupational_hazards],
        "Genetic Risk": [genetic_risk],
        "chronic Lung Disease": [chronic_lung_disease],
        "Balanced Diet": [balanced_diet],
        "Obesity": [obesity],
        "Smoking": [smoking],
        "Passive Smoker": [passive_smoker],
        "Chest Pain": [chest_pain],
        "Coughing of Blood": [coughing_of_blood],
        "Fatigue": [fatigue],
        "Weight Loss": [weight_loss],
        "Shortness of Breath": [shortness_of_breath],
        "Wheezing": [wheezing],
        "Swallowing Difficulty": [swallowing_difficulty],
        "Clubbing of Finger Nails": [clubbing_of_finger_nails],
        "Frequent Cold": [frequent_cold],
        "Dry Cough": [dry_cough],
        "Snoring": [snoring]
    })

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    level_mapping = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = level_mapping.get(prediction, "Unknown")

    st.subheader(" Prediction Results")
    st.write(f"**Predicted Risk Level:** {risk_level}")
    st.write({
        "Low": f"{prediction_proba[0]*100:.2f}%",
        "Medium": f"{prediction_proba[1]*100:.2f}%",
        "High": f"{prediction_proba[2]*100:.2f}%"
    })

    # Send to Groq for treatment recommendation
    patient_data = input_df.to_dict(orient="records")[0]
    recommendation = get_treatment_recommendation(patient_data, risk_level)

    st.subheader("💡 AI Treatment Guidance")
    st.write(recommendation)
