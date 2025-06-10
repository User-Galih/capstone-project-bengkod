# File: app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- FUNGSI UNTUK MEMUAT MODEL DAN FILE LAIN ---
# Menggunakan cache agar model tidak perlu dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_resources():
    """Memuat model, encoder, dan daftar kolom dari file lokal."""
    try:
        model = joblib.load('BengKod_Tuned_XGBoost_Model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, le, model_cols
    except FileNotFoundError as e:
        st.error(f"Error: Salah satu file model (.pkl) tidak ditemukan. Pastikan semua file ada di repository GitHub Anda. Detail: {e}")
        return None, None, None

# --- KONFIGURASI DAN PEMUATAN ---
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="wide")
model, le, model_columns = load_resources()

# --- INTERFACE APLIKASI ---
st.title("👨‍⚕️ Aplikasi Prediksi Tingkat Obesitas")
st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi tingkat obesitas berdasarkan gaya hidup dan atribut fisik.")

st.sidebar.header("Masukkan Data Anda:")

def user_input_features():
    """Membuat semua field input untuk fitur."""
    gender = st.sidebar.selectbox('Jenis Kelamin', ('Male', 'Female'))
    age = st.sidebar.slider('Umur', 14, 61, 25)
    height = st.sidebar.slider('Tinggi (meter)', 1.45, 1.98, 1.70, 0.01)
    weight = st.sidebar.slider('Berat Badan (kg)', 39.0, 173.0, 70.0, 1.0)
    family_history_with_overweight = st.sidebar.selectbox('Riwayat keluarga dengan berat badan berlebih?', ('yes', 'no'))
    favc = st.sidebar.selectbox('Sering mengonsumsi makanan tinggi kalori (FAVC)?', ('yes', 'no'))
    fcvc = st.sidebar.slider('Frekuensi konsumsi sayuran (FCVC)', 1, 3, 2)
    ncp = st.sidebar.slider('Jumlah makan utama (NCP)', 1, 4, 3)
    caec = st.sidebar.selectbox('Konsumsi makanan di antara waktu makan (CAEC)', ('no', 'Sometimes', 'Frequently', 'Always'))
    smoke = st.sidebar.selectbox('Apakah Anda merokok (SMOKE)?', ('yes', 'no'))
    ch2o = st.sidebar.slider('Konsumsi air harian (liter) (CH2O)', 1, 3, 2)
    scc = st.sidebar.selectbox('Monitor kalori makanan (SCC)?', ('yes', 'no'))
    faf = st.sidebar.slider('Frekuensi aktivitas fisik (hari/minggu) (FAF)', 0, 3, 1)
    tue = st.sidebar.slider('Waktu penggunaan perangkat teknologi (jam/hari) (TUE)', 0, 2, 1)
    calc = st.sidebar.selectbox('Konsumsi alkohol (CALC)', ('no', 'Sometimes', 'Frequently', 'Always'))
    mtrans = st.sidebar.selectbox('Moda Transportasi Utama (MTRANS)', ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'))

    data = {'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
            'family_history_with_overweight': family_history_with_overweight, 'FAVC': favc,
            'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o,
            'SCC': scc, 'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- PREPROCESSING DATA INPUT ---
if model_columns is not None:
    processed_df = pd.get_dummies(input_df, drop_first=False)
    final_df = pd.DataFrame(columns=model_columns).fillna(0)
    final_df, _ = final_df.align(processed_df, join='right', axis=1, fill_value=0)
    final_df = final_df[model_columns]

# --- TAMPILKAN DATA INPUT & PREDIKSI ---
st.subheader("Ringkasan Data Input Anda")
st.write(input_df)

if st.button("🔮 Prediksi Tingkat Obesitas"):
    if all(v is not None for v in [model, le, model_columns]):
        prediction_encoded = model.predict(final_df)
        prediction = le.inverse_transform(prediction_encoded)
        st.subheader("🎉 Hasil Prediksi Anda")
        st.success(f"**{prediction[0]}**")