# File: app.py

import streamlit as st
import pandas as pd
import joblib

# --- FUNGSI UNTUK MEMUAT SUMBER DAYA ---
@st.cache_resource
def load_resources():
    """Memuat semua file .pkl yang dibutuhkan untuk prediksi."""
    try:
        model = joblib.load('BengKod_Tuned_XGBoost_Model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, le, model_cols
    except FileNotFoundError as e:
        st.error(f"Error: Salah satu file .pkl tidak ditemukan. Pastikan semua file ada di repository. Detail: {e}")
        return None, None, None

# --- KONFIGURASI DAN PEMUATAN ---
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="centered")
model, le, model_columns = load_resources()

# --- INTERFACE APLIKASI ---
st.title("👨‍⚕️ Prediksi Tingkat Obesitas")
st.write("Aplikasi ini menggunakan model Machine Learning yang telah dilatih dengan fitur-fitur pilihan untuk memberikan prediksi yang lebih akurat.")

st.sidebar.header("Isi Data Anda:")

def user_input_features():
    """Membuat field input HANYA untuk fitur yang relevan dengan model."""
    
    st.sidebar.subheader("Atribut Utama")
    age = st.sidebar.slider('Umur Anda', 14, 65, 25)
    weight = st.sidebar.number_input('Berat Badan Anda (dalam kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.5)

    st.sidebar.subheader("Kebiasaan dan Gaya Hidup")
    family_history = st.sidebar.selectbox('Apakah ada riwayat obesitas di keluarga?', ('yes', 'no'))
    faf = st.sidebar.slider('Frekuensi aktivitas fisik (hari per minggu)', 0, 3, 1, help="0: Tidak ada, 1: 1-2 hari, 2: 2-4 hari, 3: 4-5 hari")
    ncp = st.sidebar.slider('Jumlah makan utama per hari', 1, 4, 3)
    caec = st.sidebar.selectbox('Konsumsi makanan di antara waktu makan (CAEC)', ('no', 'Sometimes', 'Frequently', 'Always'))
    calc = st.sidebar.selectbox('Frekuensi konsumsi alkohol (CALC)', ('no', 'Sometimes', 'Frequently'))

    # Membuat dictionary dari input mentah pengguna
    data = {
        'Age': age,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAF': faf,
        'NCP': ncp,
        'CAEC': caec,
        'CALC': calc
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- TAMPILKAN DATA INPUT DAN PREDIKSI ---
st.subheader("Ringkasan Data yang Anda Masukkan")
st.write(input_df)

if st.button("🔮 Prediksi Sekarang"):
    if all(v is not None for v in [model, le, model_columns]):
        # Tahap Preprocessing yang sama persis seperti di notebook
        # 1. Lakukan One-Hot Encoding pada input pengguna
        processed_input = pd.get_dummies(input_df, drop_first=False)
        
        # 2. Buat DataFrame 'template' kosong dengan semua kolom yang diharapkan model
        final_df = pd.DataFrame(columns=model_columns).fillna(0)
        
        # 3. Selaraskan input pengguna dengan template.
        # Kolom yang cocok akan diisi, yang tidak ada di input akan tetap 0.
        final_df, _ = final_df.align(processed_input, join='left', axis=1, fill_value=0)
        
        # 4. Pastikan urutan kolom sudah 100% benar
        final_df = final_df[model_columns]

        # Lakukan Prediksi
        prediction_encoded = model.predict(final_df)
        prediction_text = le.inverse_transform(prediction_encoded)[0]
        
        st.subheader("🎉 Hasil Prediksi Anda")
        st.success(f"Berdasarkan data yang Anda berikan, tingkat obesitas Anda diprediksi sebagai: **{prediction_text.replace('_', ' ')}**")

        st.info("Catatan: Prediksi ini dibuat berdasarkan model statistik dan tidak menggantikan konsultasi medis profesional.")
