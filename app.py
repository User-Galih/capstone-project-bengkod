# File: app.py (Versi Final Multi-Halaman dengan Aset Gambar)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dasbor Prediksi Obesitas",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT SUMBER DAYA ---
@st.cache_resource
def load_resources():
    """Memuat model, encoder, daftar kolom, dan data yang sudah dibersihkan."""
    try:
        model = joblib.load('model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        df_encoded = pd.read_csv('data_encoded.csv')
        return model, le, model_cols, df_encoded
    except FileNotFoundError as e:
        st.error(f"Error memuat file: {e}. Pastikan file 'model.pkl', 'label_encoder.pkl', 'model_columns.pkl', dan 'data_encoded.csv' ada di repository Anda.")
        return None, None, None, None

# --- MEMUAT SEMUA SUMBER DAYA ---
model, le, model_columns, df_encoded = load_resources()

# =====================================================================================
# --- HALAMAN UTAMA (PREDIKSI) ---
# =====================================================================================
def show_prediction_page():
    st.title("👨‍⚕️ Prediksi Tingkat Obesitas")
    st.write("Aplikasi ini menggunakan model Machine Learning (XGBoost) untuk memprediksi tingkat obesitas Anda. Silakan isi semua data di bawah ini untuk mendapatkan hasil yang akurat.")
    
    with st.form("prediction_form"):
        # Menggunakan expander untuk mengelompokkan input
        with st.expander("🧍‍♂️ **Data Diri & Fisik**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox('Jenis Kelamin (Gender)', ('Male', 'Female'), key='gender')
                age = st.slider('Umur (Age)', 14, 65, 25, key='age')
            with col2:
                height = st.number_input('Tinggi (Height, dalam meter)', 1.40, 2.10, 1.70, format="%.2f", key='height')
                weight = st.number_input('Berat Badan (Weight, dalam kg)', 30.0, 200.0, 70.0, step=0.5, key='weight')
            family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ('yes', 'no'), key='family')

        with st.expander("🍎 **Pola Makan & Minum**"):
            col3, col4 = st.columns(2)
            with col3:
                favc = st.selectbox('Sering mengonsumsi makanan tinggi kalori (FAVC)?', ('yes', 'no'), key='favc')
                ncp = st.slider('Jumlah makan utama per hari (NCP)', 1, 4, 3, key='ncp')
                scc = st.selectbox('Monitor kalori makanan (SCC)?', ('yes', 'no'), key='scc')
            with col4:
                fcvc = st.slider('Frekuensi konsumsi sayuran (FCVC)', 1, 3, 2, key='fcvc')
                st.caption("1: Tidak pernah, 2: Kadang-kadang, 3: Selalu")
                ch2o = st.slider('Konsumsi air harian (liter) (CH2O)', 1, 3, 2, key='ch2o')
                st.caption("1: <1 Liter, 2: 1-2 Liter, 3: >2 Liter")
            caec = st.selectbox('Konsumsi makanan di antara waktu makan (CAEC)', ('no', 'Sometimes', 'Frequently', 'Always'), key='caec')

        with st.expander("🏃‍♂️ **Aktivitas & Gaya Hidup**"):
            col5, col6 = st.columns(2)
            with col5:
                smoke = st.selectbox('Apakah Anda merokok (SMOKE)?', ('yes', 'no'), key='smoke')
                faf = st.slider('Frekuensi aktivitas fisik (FAF)', 0, 3, 1, key='faf')
                st.caption("0: Tidak ada, 1: 1-2 hari, 2: 2-4 hari")
            with col6:
                calc = st.selectbox('Frekuensi konsumsi alkohol (CALC)', ('no', 'Sometimes', 'Frequently', 'Always'), key='calc')
                tue = st.slider('Waktu penggunaan perangkat teknologi (TUE)', 0, 2, 1, key='tue')
                st.caption("0: 0-2 jam, 1: 3-5 jam, 2: >5 jam")
            mtrans = st.selectbox('Moda Transportasi Utama (MTRANS)', ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'), key='mtrans')

        submitted = st.form_submit_button('**Prediksi Tingkat Obesitas Saya**', use_container_width=True)

        if submitted:
            if all(v is not None for v in [model, le, model_columns]):
                data = {'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o, 
                        'FAF': faf, 'TUE': tue, 'Gender': gender, 'family_history_with_overweight': family_history,
                        'FAVC': favc, 'CAEC': caec, 'SMOKE': smoke, 'SCC': scc, 'CALC': calc, 'MTRANS': mtrans}
                input_df = pd.DataFrame([data])
                processed_input = pd.get_dummies(input_df, drop_first=False)
                final_df = processed_input.reindex(columns=model_columns, fill_value=0)
                final_numpy = final_df.to_numpy()
                prediction_encoded = model.predict(final_numpy)
                prediction_text = le.inverse_transform(prediction_encoded)[0]
                
                st.divider()
                st.metric(label="**Hasil Prediksi Tingkat Obesitas Anda**", value=prediction_text.replace('_', ' '))
                st.balloons()

# =====================================================================================
# --- HALAMAN EKSPLORASI DATA (VISUALISASI) ---
# =====================================================================================
def show_eda_page():
    st.title("📊 Eksplorasi & Visualisasi Data")
    st.write("Memahami data adalah langkah pertama dalam setiap proyek data science. Di halaman ini, kita akan melihat beberapa wawasan kunci dari dataset Obesitas.")

    if df_encoded is not None:
        # Men-decode target untuk visualisasi yang lebih baik
        df_display = df_encoded.copy()
        df_display['NObeyesdad_str'] = le.inverse_transform(df_encoded['NObeyesdad'])

        st.subheader("1. Distribusi Kategori Obesitas (Target)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(y=df_display['NObeyesdad_str'], ax=ax1, palette='viridis', order=df_display['NObeyesdad_str'].value_counts().index)
        st.pyplot(fig1)
        st.caption("Visualisasi ini menunjukkan distribusi dari variabel target kita. Kita bisa melihat bahwa datanya cukup seimbang antar kelas.")

        st.subheader("2. Distribusi Umur dan Berat Badan")
        col1, col2 = st.columns(2)
        with col1:
            fig2, ax2 = plt.subplots()
            sns.histplot(df_encoded['Age'], ax=ax2, kde=True, bins=20, color='skyblue')
            ax2.set_title('Distribusi Umur')
            st.pyplot(fig2)
        with col2:
            fig3, ax3 = plt.subplots()
            sns.histplot(df_encoded['Weight'], ax=ax3, kde=True, bins=20, color='salmon')
            ax3.set_title('Distribusi Berat Badan')
            st.pyplot(fig3)
        st.caption("Histogram ini menunjukkan bahwa mayoritas responden berada di rentang umur 20-30 tahun.")
        
    else:
        st.warning("Data untuk visualisasi tidak dapat dimuat.")

# =====================================================================================
# --- HALAMAN PERFORMA MODEL ---
# =====================================================================================
def show_model_performance_page():
    st.title("📈 Performa & Evaluasi Model")
    st.write("Bagian ini menunjukkan seberapa baik performa model XGBoost yang telah dilatih dan dioptimalkan, berdasarkan pengujian dengan data yang belum pernah dilihat sebelumnya.")

    try:
        st.subheader("1. Laporan Klasifikasi (Classification Report)")
        st.write("Laporan ini memberikan rincian metrik performa (Precision, Recall, F1-score) untuk setiap kategori obesitas.")
        st.image('classification_report.png', caption='Hasil Uji Performa Model XGBoost')

        st.divider()

        st.subheader("2. Pentingnya Fitur (Feature Importance)")
        st.write("Visualisasi ini menunjukkan fitur mana yang paling berpengaruh bagi model dalam membuat keputusan prediksi. Semakin tinggi nilainya, semakin penting fitur tersebut.")
        st.image('feature_importance.png', caption='Peringkat Fitur berdasarkan Kontribusinya pada Model')
    except FileNotFoundError:
        st.error("Satu atau kedua file gambar ('classification_report.png', 'feature_importance.png') tidak ditemukan di repository GitHub Anda. Harap upload kedua gambar tersebut.")

# =====================================================================================
# --- NAVIGASI SIDEBAR ---
# =====================================================================================
st.sidebar.title("Navigasi Dasbor")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Prediksi", "📊 Eksplorasi Data", "📈 Performa Model"])

if page == "🏠 Prediksi":
    show_prediction_page()
elif page == "📊 Eksplorasi Data":
    show_eda_page()
elif page == "📈 Performa Model":
    show_model_performance_page()
