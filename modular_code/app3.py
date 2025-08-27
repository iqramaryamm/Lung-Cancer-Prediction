# Imports 
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from sklearn.metrics import confusion_matrix
from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import train_models
from model_evaluator import evaluate_models


# ---- Config ----
st.set_page_config(page_title="Lung Cancer ML Pipeline & Risk Predictor", layout="wide")
st.title("Lung Cancer Prediction, Analysis & AI Treatment Guidance")

# ---- Load Pre-trained Model ----
model = joblib.load("Logistic_Regression.pkl")

# ---- Groq API ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = "Enter API key"  # ideally use env var
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_treatment_recommendation(patient_data: dict, risk_level: str) -> str:
    """Send patient data and risk to Groq LLM for treatment recommendation."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    prompt = f"""
    Patient details: {patient_data}
    Predicted lung cancer risk: {risk_level}.

    As a medical assistant (not a doctor), provide general treatment guidance,
    lifestyle advice, and possible next steps. Keep the response clear and empathetic.
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


# ---- Sidebar Navigation ----
app_mode = st.sidebar.radio("Choose Mode", ["Upload & Train Pipeline", "Patient Risk Predictor"])


#  MODE 1: Full ML Pipeline 
if app_mode == "Upload & Train Pipeline":
    uploaded_file = st.file_uploader("Upload your Excel dataset", type=[".xlsx"])

    if uploaded_file:
        st.success("File uploaded successfully!")

        # Step 1: Load & Prep
        st.subheader("Step 1: Load & Prepare Data")
        loader = DataLoader(target_col="Level", id_col="Patient Id")
        try:
            df, X_train, X_test, y_train, y_test = loader.load_and_prepare(uploaded_file)
            st.write("Data loaded and prepared.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error while loading data: {e}")
            st.stop()

        # Step 2: EDA
        st.subheader("Step 2: Exploratory Data Analysis")
        analyser = DataAnalyser(target_col="Level")

        with st.expander("Data Analysis Summary"):
            st.json(analyser.analyze_data(df))

        with st.expander("Data Description"):
            st.dataframe(analyser.describe(df).head())

        with st.expander("Class Balance"):
            st.write(analyser.class_balance(df))

        st.markdown("### Visualizations")
        if st.checkbox("Correlation Heatmap"):
            fig = analyser.correlation_heatmap(df)
            st.pyplot(fig)

        if st.checkbox("Histograms by Target(Level)"):
            fig = analyser.hist_by_target(df)
            st.pyplot(fig)

        # Step 3: Train Models
        st.subheader("Step 3: Training Models")
        with st.spinner("Training models..."):
            models = train_models(X_train, y_train, use_grid_search=False, cv=3)
        st.success("Models are trained.")

        # Step 4: Evaluate Models
        st.subheader("Step 4: Evaluation of Models")
        results_df = evaluate_models(models, X_test, y_test)
        st.dataframe(results_df)

        # Step 5: Confusion Matrices
        st.subheader("Step 5: Confusion Matrices")
        model_items = list(models.items())
        for i in range(0, len(model_items), 2):
            cols = st.columns(2)
            for j, (name, model) in enumerate(model_items[i:i+2]):
                with cols[j]:
                    st.markdown(f"**{name}**")
                    y_pred = model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=np.unique(y_test),
                                yticklabels=np.unique(y_test), ax=ax, cbar=False)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix - {name}", fontsize=10)
                    st.pyplot(fig)

    else:
        st.info("Upload a `.xlsx` file to begin.")


# ---- MODE 2: Patient Form Prediction ----
elif app_mode == "Patient Risk Predictor":
    st.subheader("Patient Information Form")

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
        passive_smoker = st.slider("Passive Smoker", 0, 10, 5)
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

        st.subheader("Prediction Results")
        st.write(f"Predicted Risk Level: {risk_level}")
        st.write({
            "Low": f"{prediction_proba[0]*100:.2f}%",
            "Medium": f"{prediction_proba[1]*100:.2f}%",
            "High": f"{prediction_proba[2]*100:.2f}%"
        })

        # Send to Groq for treatment recommendation
        patient_data = input_df.to_dict(orient="records")[0]
        recommendation = get_treatment_recommendation(patient_data, risk_level)

        st.subheader("AI Treatment Guidance")
        st.write(recommendation)
