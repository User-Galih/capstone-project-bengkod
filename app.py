# File: app.py (Versi yang Diperbarui)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dasbor Prediksi Obesitas",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT SUMBER DAYA ---
@st.cache_resource
def load_full_model_resources():
    """Memuat model LENGKAP dan aset-asetnya."""
    try:
        model = joblib.load('BengKod_Default_XGBoost_Model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        df_encoded = pd.read_csv('data_encoded.csv')
        return model, le, model_cols, df_encoded
    except FileNotFoundError as e:
        st.error(f"Error memuat file model lengkap: {e}.")
        return None, None, None, None

@st.cache_resource
def load_simple_model_resources():
    """Memuat model RINGKAS dan aset-asetnya."""
    try:
        simple_model = joblib.load('Model_Ringkas_XGBoost.pkl')
        simple_model_cols = joblib.load('model_columns_ringkas.pkl')
        return simple_model, simple_model_cols
    except FileNotFoundError as e:
        st.error(f"Error memuat file model ringkas: {e}.")
        return None, None

# --- MEMUAT SEMUA SUMBER DAYA ---
full_model, le, full_model_columns, df_encoded = load_full_model_resources()
simple_model, simple_model_columns = load_simple_model_resources()

# --- FUNGSI UNTUK PEMETAAN MANUAL (HARDCODING) ---
# Ini untuk mengubah input teks dari pengguna menjadi angka sesuai training
def map_inputs(input_df):
    # Salin untuk menghindari mengubah DataFrame asli
    df = input_df.copy()
    
    # Pemetaan untuk semua fitur kategorikal
    gender_map = {'Female': 0, 'Male': 1}
    yes_no_map = {'no': 0, 'yes': 1}
    caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    mtrans_map = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}

    df['Gender'] = df['Gender'].map(gender_map)
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    df['FAVC'] = df['FAVC'].map(yes_no_map)
    df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    df['SCC'] = df['SCC'].map(yes_no_map)
    df['CAEC'] = df['CAEC'].map(caec_map)
    df['CALC'] = df['CALC'].map(calc_map)
    df['MTRANS'] = df['MTRANS'].map(mtrans_map)
    
    return df

# =====================================================================================
# --- FUNGSI UNTUK SETIAP HALAMAN ---
# =====================================================================================

def show_full_prediction_page():
    st.title("👨‍⚕️ Prediksi Lengkap (Model XGBoost)")
    st.info("Gunakan semua fitur gaya hidup untuk mendapatkan prediksi yang paling komprehensif dari model Machine Learning.", icon="💡")
    
    with st.form("prediction_form"):
        st.header("Isi Data Diri Anda")
        # --- Input fields (sama seperti kode lama Anda) ---
        with st.expander("🧍‍♂️ **Data Diri & Fisik**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
                age = st.slider('Umur', 14, 65, 25)
            with col2:
                height = st.number_input('Tinggi (meter)', 1.40, 2.20, 1.70, format="%.2f")
                weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0, step=0.5)
            family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ['yes', 'no'])

        with st.expander("🍎 **Pola Makan & Minum**"):
            # ... (semua input lainnya seperti kode lama Anda) ...
            favc = st.selectbox('Sering konsumsi makanan tinggi kalori?', ['yes', 'no'])
            fcvc = st.slider('Frekuensi konsumsi sayuran (1-3)', 1, 3, 2)
            ncp = st.slider('Jumlah makan utama per hari', 1, 4, 3)
            caec = st.selectbox('Konsumsi makanan di antara waktu makan', ['no', 'Sometimes', 'Frequently', 'Always'])
            ch2o = st.slider('Konsumsi air harian (1-3 liter)', 1, 3, 2)
            scc = st.selectbox('Monitor kalori makanan?', ['yes', 'no'])

        with st.expander("🏃‍♂️ **Aktivitas & Gaya Hidup**"):
            # ... (semua input lainnya seperti kode lama Anda) ...
            smoke = st.selectbox('Apakah Anda merokok?', ['yes', 'no'])
            faf = st.slider('Frekuensi aktivitas fisik (0-3 hari/minggu)', 0, 3, 1)
            tue = st.slider('Waktu penggunaan gadget (0-2 jam/hari)', 0, 2, 1)
            calc = st.selectbox('Frekuensi konsumsi alkohol', ['no', 'Sometimes', 'Frequently', 'Always'])
            mtrans = st.selectbox('Transportasi Utama', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

        submitted = st.form_submit_button('**Prediksi Sekarang**', use_container_width=True, type="primary")

        if submitted:
            if all(v is not None for v in [full_model, le, full_model_columns]):
                data = {'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o, 
                        'FAF': faf, 'TUE': tue, 'Gender': gender, 'family_history_with_overweight': family_history,
                        'FAVC': favc, 'CAEC': caec, 'SMOKE': smoke, 'SCC': scc, 'CALC': calc, 'MTRANS': mtrans}
                
                input_df = pd.DataFrame([data])
                processed_input = map_inputs(input_df) # <-- LOGIKA BARU
                final_df = processed_input.reindex(columns=full_model_columns, fill_value=0) # <-- LOGIKA BARU
                
                prediction_encoded = full_model.predict(final_df)
                prediction_text = le.inverse_transform(prediction_encoded)[0]
                
                st.divider()
                st.metric(label="**Hasil Prediksi Tingkat Obesitas Anda**", value=prediction_text.replace('_', ' '))
                st.balloons()

def show_simple_prediction_page():
    st.title("⚡ Prediksi Ringkas (5 Fitur Utama)")
    st.info("Dapatkan prediksi cepat hanya dengan 5 fitur kunci yang paling berpengaruh.", icon="💡")
    
    with st.form("simple_prediction_form"):
        st.header("Isi 5 Data Kunci")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0, step=0.5)
            height = st.number_input('Tinggi (meter)', 1.40, 2.20, 1.70, format="%.2f")
            age = st.slider('Umur', 14, 65, 25)
        with col2:
            gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
            family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ['yes', 'no'])

        submitted = st.form_submit_button('**Prediksi Cepat**', use_container_width=True, type="primary")

        if submitted:
            if all(v is not None for v in [simple_model, le, simple_model_columns]):
                data = {'Weight': weight, 'Height': height, 'Age': age, 'Gender': gender, 'family_history_with_overweight': family_history}
                
                input_df = pd.DataFrame([data])
                processed_input = map_inputs(input_df) # <-- LOGIKA BARU
                final_df = processed_input.reindex(columns=simple_model_columns, fill_value=0) # <-- LOGIKA BARU
                
                prediction_encoded = simple_model.predict(final_df)
                prediction_text = le.inverse_transform(prediction_encoded)[0]
                
                st.divider()
                st.metric(label="**Hasil Prediksi Tingkat Obesitas Anda**", value=prediction_text.replace('_', ' '))
                st.snow()

def show_eda_page():
    # ... (kode untuk halaman EDA tidak berubah) ...
    st.title("📊 Eksplorasi & Visualisasi Data")
    st.write("Halaman ini menampilkan beberapa wawasan kunci dari dataset Obesitas.")
    if df_encoded is not None and le is not None:
        # Men-decode target untuk visualisasi yang lebih baik
        df_display = df_encoded.copy()
        df_display['NObeyesdad_str'] = le.inverse_transform(df_encoded['NObeyesdad'])
        st.subheader("Distribusi Kategori Obesitas")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=df_display['NObeyesdad_str'], ax=ax, palette='viridis', order=df_display['NObeyesdad_str'].value_counts().index)
        st.pyplot(fig)
    else:
        st.warning("Data untuk visualisasi tidak dapat dimuat.")


def show_model_performance_page():
    # ... (kode untuk halaman Performa Model tidak berubah) ...
    st.title("📈 Performa & Evaluasi Model")
    st.write("Bagian ini menampilkan evaluasi dari model XGBoost yang digunakan.")
    try:
        st.subheader("Pentingnya Fitur (Feature Importance)")
        st.image('feature_importance.png', caption='Peringkat Fitur berdasarkan Kontribusinya pada Model')
    except FileNotFoundError:
        st.error("File gambar 'feature_importance.png' tidak ditemukan.")

# =====================================================================================
# --- NAVIGASI SIDEBAR (Tampilan Baru) ---
# =====================================================================================
st.sidebar.header("Dasbor Obesitas")

if 'page' not in st.session_state:
    st.session_state.page = 'Prediksi Lengkap'

if st.sidebar.button("👨‍⚕️ Prediksi Lengkap", use_container_width=True, type='primary' if st.session_state.page == 'Prediksi Lengkap' else 'secondary'):
    st.session_state.page = 'Prediksi Lengkap'
if st.sidebar.button("⚡ Prediksi Ringkas", use_container_width=True, type='primary' if st.session_state.page == 'Prediksi Ringkas' else 'secondary'):
    st.session_state.page = 'Prediksi Ringkas'
if st.sidebar.button("📊 Eksplorasi Data", use_container_width=True, type='primary' if st.session_state.page == 'Eksplorasi Data' else 'secondary'):
    st.session_state.page = 'Eksplorasi Data'
if st.sidebar.button("📈 Performa Model", use_container_width=True, type='primary' if st.session_state.page == 'Performa Model' else 'secondary'):
    st.session_state.page = 'Performa Model'


# --- Kontrol Halaman ---
if st.session_state.page == 'Prediksi Lengkap':
    show_full_prediction_page()
elif st.session_state.page == 'Prediksi Ringkas':
    show_simple_prediction_page()
elif st.session_state.page == 'Eksplorasi Data':
    show_eda_page()
elif st.session_state.page == 'Performa Model':
    show_model_performance_page()
