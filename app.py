# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Student Score Predictor", layout="centered")

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎓 Student Math Score Predictor")
st.write("Predict a student's math score based on their other scores and background.")

# Sidebar inputs
st.sidebar.header("Input Student Information")

reading = st.sidebar.slider("Reading Score", 0, 100, 70)
writing = st.sidebar.slider("Writing Score", 0, 100, 70)
gender = st.sidebar.selectbox("Gender", ['male', 'female'])
race = st.sidebar.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
education = st.sidebar.selectbox("Parental Education Level", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.sidebar.selectbox("Lunch Type", ['standard', 'free/reduced'])
prep = st.sidebar.selectbox("Test Preparation Course", ['none', 'completed'])

# Input dictionary
def create_input():
    return pd.DataFrame([{
        'reading score': reading,
        'writing score': writing,
        'gender_female': 1 if gender == 'female' else 0,
        'race/ethnicity_group B': 1 if race == 'group B' else 0,
        'race/ethnicity_group C': 1 if race == 'group C' else 0,
        'race/ethnicity_group D': 1 if race == 'group D' else 0,
        'race/ethnicity_group E': 1 if race == 'group E' else 0,
        "parental level of education_high school": 1 if education == "high school" else 0,
        "parental level of education_some college": 1 if education == "some college" else 0,
        "parental level of education_some high school": 1 if education == "some high school" else 0,
        "parental level of education_bachelor's degree": 1 if education == "bachelor's degree" else 0,
        "parental level of education_master's degree": 1 if education == "master's degree" else 0,
        "lunch_standard": 1 if lunch == "standard" else 0,
        "test preparation course_none": 1 if prep == "none" else 0,
    }])

input_df = create_input()

# Prediction
if st.button("📊 Predict Math Score"):
    result = model.predict(input_df)[0]
    st.success(f"✅ Predicted Math Score: {result:.2f}")
    st.balloons()
