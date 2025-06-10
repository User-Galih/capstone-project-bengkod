# File: app.py (Versi Final dengan Desain UI/UX Baru)

import streamlit as st
import pandas as pd
import joblib

# --- FUNGSI UNTUK MEMUAT SUMBER DAYA ---
@st.cache_resource
def load_resources():
    """Memuat semua file .pkl yang dibutuhkan untuk prediksi."""
    try:
        # Pastikan nama file ini sesuai dengan yang ada di repository Anda
        model = joblib.load('model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, le, model_cols
    except FileNotFoundError:
        st.error("Satu atau lebih file .pkl tidak ditemukan. Pastikan semua file (model.pkl, label_encoder.pkl, model_columns.pkl) ada di repository.")
        return None, None, None

# --- KONFIGURASI DAN PEMUATAN ---
st.set_page_config(page_title="Prediksi Obesitas", layout="centered", initial_sidebar_state="auto")
model, le, model_columns = load_resources()

# --- INTERFACE APLIKASI ---
st.title("👨‍⚕️ Prediksi Tingkat Obesitas")
st.write("Aplikasi ini menggunakan model Machine Learning (XGBoost) untuk memprediksi tingkat obesitas Anda. Silakan isi semua data di bawah ini untuk mendapatkan hasil yang akurat.")

# --- FORM INPUT DENGAN DESAIN BARU ---

# Menggunakan expander untuk mengelompokkan input
with st.expander("🧍‍♂️ **Data Diri & Fisik**", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Jenis Kelamin (Gender)', ('Male', 'Female'))
        age = st.slider('Umur (Age)', 14, 65, 25)
    with col2:
        height = st.number_input('Tinggi (Height, dalam meter)', 1.40, 2.10, 1.70, format="%.2f")
        weight = st.number_input('Berat Badan (Weight, dalam kg)', 30.0, 200.0, 70.0, step=0.5)
    
    family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih (family_history_with_overweight)?', ('yes', 'no'))

with st.expander("🍎 **Pola Makan & Minum**"):
    col3, col4 = st.columns(2)
    with col3:
        favc = st.selectbox('Sering mengonsumsi makanan tinggi kalori (FAVC)?', ('yes', 'no'))
        ncp = st.slider('Jumlah makan utama per hari (NCP)', 1, 4, 3)
        scc = st.selectbox('Monitor kalori makanan (SCC)?', ('yes', 'no'))
    with col4:
        fcvc = st.slider('Frekuensi konsumsi sayuran (FCVC)', 1, 3, 2)
        st.caption("1: Tidak pernah, 2: Kadang-kadang, 3: Selalu")
        ch2o = st.slider('Konsumsi air harian (liter) (CH2O)', 1, 3, 2)
        st.caption("1: <1 Liter, 2: 1-2 Liter, 3: >2 Liter")

    caec = st.selectbox('Konsumsi makanan di antara waktu makan (CAEC)', ('no', 'Sometimes', 'Frequently', 'Always'))

with st.expander("🏃‍♂️ **Aktivitas & Gaya Hidup**"):
    col5, col6 = st.columns(2)
    with col5:
        smoke = st.selectbox('Apakah Anda merokok (SMOKE)?', ('yes', 'no'))
        faf = st.slider('Frekuensi aktivitas fisik (hari/minggu) (FAF)', 0, 3, 1)
        st.caption("0: Tidak ada, 1: 1-2 hari, 2: 2-4 hari") # Koreksi dari sebelumnya
    with col6:
        calc = st.selectbox('Frekuensi konsumsi alkohol (CALC)', ('no', 'Sometimes', 'Frequently', 'Always'))
        tue = st.slider('Waktu penggunaan perangkat teknologi (jam/hari) (TUE)', 0, 2, 1)
        st.caption("0: 0-2 jam, 1: 3-5 jam, 2: >5 jam")

    mtrans = st.selectbox('Moda Transportasi Utama (MTRANS)', ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'))


# Tombol Prediksi diletakkan di tengah bawah
if st.button('**Prediksi Tingkat Obesitas Saya**', use_container_width=True, type="primary"):
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

        # Tampilkan Hasil dengan Desain Baru
        st.divider()
        st.metric(
            label="**Hasil Prediksi Tingkat Obesitas Anda**",
            value=prediction_text.replace('_', ' ')
        )
        st.balloons()
