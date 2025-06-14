import streamlit as st
import joblib
import numpy as np

# Load model dan komponen
model = joblib.load("Assets/model_xgb_tuned.pkl")
scaler = joblib.load("Assets/scaler.pkl")
selected_features = joblib.load("Assets/selected_features.pkl")

# Mapping label ke deskripsi kelas
label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

# Konfigurasi halaman
st.set_page_config(page_title="BodyFit Classifier", page_icon="🧠", layout="centered")

# Tampilan header dan deskripsi
st.markdown("""
    <style>
    .header-title {
        font-size: 40px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
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
    <div class="header-title">🧠 BodyFit Classifier</div>
    <div class="sub-header">Smart Prediction for Your Body Weight Category Based on Health Features</div>
""", unsafe_allow_html=True)

# Input kolom
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=74.0, step=0.1)

with col2:
    age = st.number_input("Age", min_value=3, max_value=98, value=22, step=1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=174.0, step=0.1)

fam_history = st.radio("Family history of obesity?", ["Yes", "No"], horizontal=True)

# Tombol prediksi
if st.button("🎯 Predict My Category"):
    # Encode input
    gender_encoded = 1 if gender == "Male" else 0
    fam_history_encoded = 1 if fam_history == "Yes" else 0
    height_m = height / 100  # convert cm to meter

    # Siapkan input
    input_data = np.array([[age, gender_encoded, height_m, weight, fam_history_encoded]])
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)
    result_label = label_mapping.get(int(prediction[0]), "Unknown")

    # Tampilkan hasil
    st.markdown("---")
    st.markdown("<div class='result-title'>Your Body Weight Category:</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value'>{result_label}</div>", unsafe_allow_html=True)

# Penjelasan proyek
st.markdown("---")
st.markdown("## 📘 Project Background")
st.markdown("""
Berdasarkan hasil seleksi fitur menggunakan metode **ANOVA F-score**, saya memilih 5 fitur teratas yang memiliki korelasi paling tinggi terhadap label klasifikasi, yaitu:

- **Weight (Berat Badan)**
- **Gender (Jenis Kelamin)**
- **family_history_with_overweight (Riwayat Keluarga yang Kelebihan Berat Badan)**
- **FCVC (Frekuensi Konsumsi Sayur)**
- **Age (Usia)**

Namun, demi kemudahan dan relevansi dalam proses deployment, saya mengganti fitur **FCVC** dengan **Height (Tinggi Badan)**. Keputusan ini diambil untuk menyesuaikan dengan pendekatan yang digunakan oleh organisasi seperti **WHO** dan metode **Body Mass Index (BMI)**, yang mempertimbangkan berat badan dan tinggi badan sebagai indikator utama untuk mengukur status gizi seseorang.
""")

# Referensi
st.markdown("### 🔗 References")
st.markdown("""
- [BMI Calculator – Calculator.net](https://www.calculator.net/bmi-calculator.html)  
- [Truth About Weight – Global Initiative](https://www.truthaboutweight.global/)
""")
