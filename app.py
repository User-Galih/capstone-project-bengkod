# File: app.py (Versi Final untuk SEMUA Fitur)

import streamlit as st
import pandas as pd
import joblib

# --- FUNGSI UNTUK MEMUAT SUMBER DAYA ---
@st.cache_resource
def load_resources():
    """Memuat semua file .pkl yang dibutuhkan untuk prediksi."""
    try:
        # Ganti nama file ini jika Anda menyimpannya dengan nama berbeda
        model = joblib.load('BengKod_Tuned_XGBoost_Model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, le, model_cols
    except FileNotFoundError as e:
        st.error(f"Error: Salah satu file .pkl tidak ditemukan. Pastikan semua file (model.pkl, label_encoder.pkl, model_columns.pkl) ada di repository. Detail: {e}")
        return None, None, None

# --- KONFIGURASI DAN PEMUATAN ---
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="wide")
model, le, model_columns = load_resources()

# --- INTERFACE APLIKASI ---
st.title("👨‍⚕️ Prediksi Tingkat Obesitas")
st.write("Aplikasi ini menggunakan model Machine Learning (XGBoost) untuk memprediksi tingkat obesitas berdasarkan riwayat kesehatan, gaya hidup, dan atribut fisik.")

# --- Membuat Kolom Input agar lebih rapi ---
col1, col2 = st.columns(2)

with col1:
    st.header("Atribut Personal & Utama")
    gender = st.selectbox('Jenis Kelamin (Gender)', ('Male', 'Female'))
    age = st.slider('Umur (Age)', 14, 65, 25)
    height = st.number_input('Tinggi (Height, dalam meter)', 1.40, 2.10, 1.70, format="%.2f")
    weight = st.number_input('Berat Badan (Weight, dalam kg)', 30.0, 200.0, 70.0, step=0.5)
    family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih (family_history_with_overweight)?', ('yes', 'no'))

with col2:
    st.header("Kebiasaan & Gaya Hidup")
    favc = st.selectbox('Sering mengonsumsi makanan tinggi kalori (FAVC)?', ('yes', 'no'))
    fcvc = st.slider('Frekuensi konsumsi sayuran (FCVC)', 1, 3, 2, help="1: Tidak pernah, 2: Kadang-kadang, 3: Selalu")
    ncp = st.slider('Jumlah makan utama per hari (NCP)', 1, 4, 3)
    caec = st.selectbox('Konsumsi makanan di antara waktu makan (CAEC)', ('no', 'Sometimes', 'Frequently', 'Always'))
    smoke = st.selectbox('Apakah Anda merokok (SMOKE)?', ('yes', 'no'))
    ch2o = st.slider('Konsumsi air harian (liter) (CH2O)', 1, 3, 2, help="1: <1L, 2: 1-2L, 3: >2L")
    scc = st.selectbox('Monitor kalori makanan (SCC)?', ('yes', 'no'))
    faf = st.slider('Frekuensi aktivitas fisik (hari/minggu) (FAF)', 0, 3, 1, help="0: Tidak ada, 1: 1-2 hari, 2: 2-4 hari")
    tue = st.slider('Waktu penggunaan perangkat teknologi (jam/hari) (TUE)', 0, 2, 1, help="0: 0-2 jam, 1: 3-5 jam, 2: >5 jam")
    calc = st.selectbox('Frekuensi konsumsi alkohol (CALC)', ('no', 'Sometimes', 'Frequently', 'Always'))
    mtrans = st.selectbox('Moda Transportasi Utama (MTRANS)', ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'))

# Tombol Prediksi diletakkan di tengah bawah
if st.button('**Prediksi Tingkat Obesitas Saya**', use_container_width=True):
    if all(v is not None for v in [model, le, model_columns]):
        # Membuat dictionary dari SEMUA input pengguna
        data = {
            'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp,
            'CH2O': ch2o, 'FAF': faf, 'TUE': tue, 'Gender': gender, 'family_history_with_overweight': family_history,
            'FAVC': favc, 'CAEC': caec, 'SMOKE': smoke, 'SCC': scc, 'CALC': calc, 'MTRANS': mtrans
        }
        input_df = pd.DataFrame([data])

        # Preprocessing input agar cocok dengan model
        processed_input = pd.get_dummies(input_df, drop_first=False)
        final_df = processed_input.reindex(columns=model_columns, fill_value=0)
        final_numpy = final_df.to_numpy()

        # Lakukan Prediksi
        prediction_encoded = model.predict(final_numpy)
        prediction_text = le.inverse_transform(prediction_encoded)[0]

        # Tampilkan Hasil
        st.success(f"## Hasil Prediksi: **{prediction_text.replace('_', ' ')}**")
        st.balloons()
