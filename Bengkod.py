import streamlit as st
import joblib
import numpy as np

# Load model 
model = joblib.load("Assets/model_xgb_tuned.pkl")
scaler = joblib.load("Assets/scaler.pkl")
selected_features = joblib.load("Assets/selected_features.pkl")

label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}


st.markdown("""
    <style>
    .title-container {
        margin-bottom: 25px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    <div class="title-container">
        <h1 style='text-align: center;'>💡 Are We Obesity?</h1>
        <p style='text-align: center;'>Input your data to predict your weight category.</p>
    </div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=74.0, step=0.1)

with col2:
    age = st.number_input("Age", min_value=3, max_value=98, value=22, step=1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=174.0, step=0.1)

fam_history = st.radio("Family history of obesity?", ["Yes", "No"], horizontal=True)

# Predict button
if st.button("🔍 Predict My Weight Category"):
    # Encode input
    gender_encoded = 1 if gender == "Male" else 0
    fam_history_encoded = 1 if fam_history == "Yes" else 0
    height_m = height / 100  
    input_data = np.array([[age, gender_encoded, height_m, weight, fam_history_encoded]])
    input_scaled = scaler.transform(input_data)
    
    
    prediction = model.predict(input_scaled)
    result_label = label_mapping.get(int(prediction[0]), "Unknown")
    
    
    st.markdown("---")
    st.markdown("Your Category is..")
    st.markdown(f"<h2>{result_label}</h2>", unsafe_allow_html=True)
