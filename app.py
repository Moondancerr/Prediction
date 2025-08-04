import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

st.title("🎓 Student Math Score Predictor (No Pickle, Trains Live)")

# Load CSV from URL (you can change to local file if you want)
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv"

@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()
st.write("Sample data:", data.head())

# Preprocessing function
def preprocess(df):
    X = df.drop("math score", axis=1)
    y = df["math score"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y

X, y = preprocess(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Input widgets
st.sidebar.header("Input Student Info")

def user_input_features():
    reading = st.sidebar.slider("Reading Score", 0, 100, 70)
    writing = st.sidebar.slider("Writing Score", 0, 100, 70)
    gender = st.sidebar.selectbox("Gender", ["female", "male"])
    race = st.sidebar.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
    parent_edu = st.sidebar.selectbox("Parental Level of Education", [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
    prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

    input_dict = {
        "reading score": reading,
        "writing score": writing,
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch,
        "test preparation course": prep,
    }
    return pd.DataFrame([input_dict])

input_df = user_input_features()

# Preprocess input (one-hot encoding)
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Add missing columns to input_encoded (to match training data)
for col in X.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match training data
input_encoded = input_encoded[X.columns]

# Predict button
if st.button("Predict Math Score"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Predicted Math Score: {prediction:.2f}")
