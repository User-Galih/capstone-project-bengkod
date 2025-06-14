import streamlit as st
import joblib
import numpy as np

# Load model dan asset
model = joblib.load("Assets/model_xgb_tuned (2).pkl")
scaler = joblib.load("Assets/scaler (3).pkl")
selected_features = joblib.load("Assets/selected_features.pkl")
label_encoder = joblib.load("Assets/label_encoder (3).pkl")  # GUNAKAN ENCODER

# Konfigurasi halaman
st.set_page_config(page_title="BodyFit Classifier", page_icon="💪", layout="centered")

# Sidebar identitas pembuat
with st.sidebar:
    st.markdown("### 👨‍💻 Developer")
    st.markdown("Galih Putra Pratama")
    st.markdown("NIM: A11.2022.14359")
    st.markdown("---")
    st.markdown("📘 **Tentang Proyek**")
    st.markdown("""
    Aplikasi ini digunakan untuk mengklasifikasikan kategori berat badan berdasarkan data sederhana seperti usia, jenis kelamin, tinggi badan, berat badan, dan riwayat keluarga.
    
    Fitur yang dipilih mengacu pada pendekatan WHO dan BMI.
    """)
    st.markdown("📎 **Referensi**")
    st.markdown("""
    - [BMI Calculator – Calculator.net](https://www.calculator.net/bmi-calculator.html)  
    - [Truth About Weight – Global Initiative](https://www.truthaboutweight.global/)
    """)

# Judul utama dan deskripsi
st.markdown("""
    <style>
    .title-text {
        font-size: 40px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-text {
        font-size: 18px;
        color: #7f8c8d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 30px;
    }
    .result-title {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        text-align: center;
    }
    .result-value {
        font-size: 32px;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
    }
    </style>
    <div class="title-text">💪 BodyFit Classifier</div>
    <div class="sub-text">Masukkan datamu untuk mengetahui kategori berat badanmu.</div>
""", unsafe_allow_html=True)

# Input form
col1, col2 = st.columns(2)
with col1:
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
with col2:
    age = st.number_input("Age", min_value=3, max_value=98, value=25, step=1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)

fam_history = st.radio("Family history of obesity?", ["Yes", "No"], horizontal=True)

# Tombol prediksi
if st.button("🎯 Predict My Category"):
    gender_encoded = 1 if gender == "Male" else 0
    fam_encoded = 1 if fam_history == "Yes" else 0
    height_m = height / 100  # Ubah ke meter agar sesuai dengan training data

    input_data = np.array([[age, gender_encoded, height_m, weight, fam_encoded]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    result_label = label_encoder.inverse_transform(prediction)[0]  # Pakai encoder, bukan mapping manual

    st.markdown("---")
    st.markdown("<div class='result-title'>Your Body Weight Category:</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value'>{result_label}</div>", unsafe_allow_html=True)
