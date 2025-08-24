# Built a Streamlit app 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import train_models
from model_evaluator import evaluate_models

st.set_page_config(page_title="Cancer Patient ML Pipeline", layout="wide")
st.title("Lung Cancer Prediction")

uploaded_file = st.file_uploader("📁 Upload your Excel dataset", type=[".xlsx"])

if uploaded_file:
    st.success("File uploaded successfully!")

    # Step 1: Load & Prep
    st.subheader(" Step 1: Load & Prepare Data")
    loader = DataLoader(target_col="Level", id_col="Patient Id")
    try:
        df, X_train, X_test, y_train, y_test = loader.load_and_prepare(uploaded_file)
        st.write("Data loaded and prepared.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f" Error while loading data: {e}")
        st.stop()

    # Step 2: EDA
    st.subheader(" Step 2: Exploratory Data Analysis")
    analyser = DataAnalyser(target_col="Level")

    with st.expander(" Data Analysis Summary"):
        st.json(analyser.analyze_data(df))

    with st.expander(" Data Description"):
        st.dataframe(analyser.describe(df).head())

    with st.expander("Class Balance"):
        st.write(analyser.class_balance(df))

    st.markdown(" Optional Visualizations")
    if st.checkbox("Show Correlation Heatmap"):
        fig = analyser.correlation_heatmap(df)
        st.pyplot(fig)

    if st.checkbox("Show Histograms by Target"):
        fig=analyser.hist_by_target(df)
        st.pyplot(fig)


    # Step 3: Train Models
    st.subheader(" Step 3: Training Models")
    with st.spinner("Training models..."):
        models = train_models(X_train, y_train, use_grid_search=False, cv=3)
    st.success("✅ Models trained.")

    # Step 4: Evaluate Models
    st.subheader(" Step 4: Evaluation of  Models")
    results_df = evaluate_models(models, X_test, y_test)
    st.dataframe(results_df)

else:
    st.info("Upload a `.xlsx` file to begin.")
