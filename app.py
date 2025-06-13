# File: app.py (Versi Final - Dasbor Proyek Lengkap)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dasbor Proyek Obesitas",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI-FUNGSI PEMUAT SUMBER DAYA ---
@st.cache_resource
def load_full_model_resources():
    """Memuat model LENGKAP dan aset-asetnya."""
    try:
        model = joblib.load('BengKod_Default_XGBoost_Model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, le, model_cols
    except Exception as e:
        st.error(f"Gagal memuat aset model lengkap: {e}")
        return None, None, None

@st.cache_resource
def load_simple_model_resources():
    """Memuat model RINGKAS dan aset-asetnya."""
    try:
        simple_model = joblib.load('Model_Ringkas_XGBoost.pkl')
        simple_model_cols = joblib.load('model_columns_ringkas.pkl')
        return simple_model, simple_model_cols
    except Exception as e:
        st.error(f"Gagal memuat aset model ringkas: {e}")
        return None, None

# --- MEMUAT SEMUA SUMBER DAYA ---
full_model, le, full_model_columns = load_full_model_resources()
simple_model, simple_model_columns = load_simple_model_resources()

# --- FUNGSI PEMETAAN INPUT ---
def map_inputs(input_df):
    """Mengubah input teks dari pengguna menjadi angka sesuai training."""
    df = input_df.copy()
    gender_map = {'Female': 0, 'Male': 1}
    yes_no_map = {'no': 0, 'yes': 1}
    caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    mtrans_map = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}

    # Gunakan .get() untuk keamanan jika kolom tidak ada
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map(gender_map)
    if 'family_history_with_overweight' in df.columns:
        df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    if 'FAVC' in df.columns: df['FAVC'] = df['FAVC'].map(yes_no_map)
    if 'SMOKE' in df.columns: df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    if 'SCC' in df.columns: df['SCC'] = df['SCC'].map(yes_no_map)
    if 'CAEC' in df.columns: df['CAEC'] = df['CAEC'].map(caec_map)
    if 'CALC' in df.columns: df['CALC'] = df['CALC'].map(calc_map)
    if 'MTRANS' in df.columns: df['MTRANS'] = df['MTRANS'].map(mtrans_map)
    
    return df

# =====================================================================================
# --- FUNGSI UNTUK HALAMAN-HALAMAN APLIKASI ---
# =====================================================================================

def show_home_page():
    st.title("🚀 Selamat Datang di Dasbor Proyek Obesitas")
    st.markdown("""
    Aplikasi ini adalah demonstrasi lengkap dari sebuah proyek machine learning, mulai dari analisis data hingga deployment model prediktif.
    
    **Apa yang bisa Anda lakukan di sini?**
    - **Prediksi Interaktif**: Gunakan dua model berbeda untuk memprediksi tingkat obesitas berdasarkan input Anda.
    - **Jelajahi Proses Proyek**: Lihat bagaimana data dianalisis, dibersihkan, dan disiapkan untuk pemodelan.
    - **Lihat Performa Model**: Pahami seberapa baik model yang kami bangun dan fitur apa yang paling memengaruhinya.
    
    Gunakan menu navigasi di sebelah kiri untuk menjelajahi setiap bagian dari dasbor ini.
    """)
    # st.image("logo.png", width=400) # Ganti dengan nama file logo Anda jika ada

def show_full_prediction_page():
    st.title("👨‍⚕️ Prediksi Lengkap (Model XGBoost)")
    st.info("Gunakan semua fitur gaya hidup untuk mendapatkan prediksi yang paling komprehensif dari model Machine Learning.", icon="💡")
    
    with st.form("prediction_form"):
        st.header("Isi Data Diri Anda")
        with st.expander("🧍‍♂️ **Data Diri & Fisik**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'], key='full_gender')
                age = st.slider('Umur', 14, 65, 25, key='full_age')
            with col2:
                height = st.number_input('Tinggi (meter)', 1.40, 2.20, 1.70, format="%.2f", key='full_height')
                weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0, step=0.5, key='full_weight')
            family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ['yes', 'no'], key='full_family')

        with st.expander("🍎 **Pola Makan & Minum**"):
            col3, col4 = st.columns(2)
            with col3:
                favc = st.selectbox('Sering konsumsi makanan tinggi kalori?', ['yes', 'no'], key='full_favc')
                ncp = st.slider('Jumlah makan utama per hari', 1.0, 4.0, 3.0, step=0.1, key='full_ncp')
                scc = st.selectbox('Monitor kalori makanan?', ['yes', 'no'], key='full_scc')
            with col4:
                fcvc = st.slider('Frekuensi konsumsi sayuran (1: Tidak, 2: Kdd, 3: Selalu)', 1.0, 3.0, 2.0, step=0.1, key='full_fcvc')
                ch2o = st.slider('Konsumsi air harian (1: <1L, 2: 1-2L, 3: >2L)', 1.0, 3.0, 2.0, step=0.1, key='full_ch2o')
                caec = st.selectbox('Konsumsi makanan di antara waktu makan', ['no', 'Sometimes', 'Frequently', 'Always'], key='full_caec')

        with st.expander("🏃‍♂️ **Aktivitas & Gaya Hidup**"):
            col5, col6 = st.columns(2)
            with col5:
                smoke = st.selectbox('Apakah Anda merokok?', ['yes', 'no'], key='full_smoke')
                faf = st.slider('Frekuensi aktivitas fisik (0-3 hari/minggu)', 0.0, 3.0, 1.0, step=0.1, key='full_faf')
            with col6:
                tue = st.slider('Waktu penggunaan gadget (0: 0-2j, 1: 3-5j, 2: >5j)', 0.0, 2.0, 1.0, step=0.1, key='full_tue')
                calc = st.selectbox('Frekuensi konsumsi alkohol', ['no', 'Sometimes', 'Frequently'], key='full_calc') # 'Always' dihilangkan jika jarang ada
                mtrans = st.selectbox('Transportasi Utama', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'], key='full_mtrans')

        submitted = st.form_submit_button('**Prediksi Sekarang**', use_container_width=True, type="primary")

        if submitted:
            if all(v is not None for v in [full_model, le, full_model_columns]):
                data = {'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o, 
                        'FAF': faf, 'TUE': tue, 'Gender': gender, 'family_history_with_overweight': family_history,
                        'FAVC': favc, 'CAEC': caec, 'SMOKE': smoke, 'SCC': scc, 'CALC': calc, 'MTRANS': mtrans}
                
                input_df = pd.DataFrame([data])
                processed_input = map_inputs(input_df)
                final_df = processed_input.reindex(columns=full_model_columns, fill_value=0)
                
                prediction_encoded = full_model.predict(final_df)
                prediction_text = le.inverse_transform(prediction_encoded)[0]
                
                st.divider()
                st.success(f"**Hasil Prediksi:** Anda masuk dalam kategori **{prediction_text.replace('_', ' ')}**")
                st.balloons()
            else:
                st.error("Model atau aset lainnya gagal dimuat. Tidak bisa melakukan prediksi.")

def show_simple_prediction_page():
    st.title("⚡ Prediksi Ringkas (5 Fitur Utama)")
    st.info("Dapatkan prediksi cepat hanya dengan 5 fitur kunci yang paling berpengaruh.", icon="💡")
    
    with st.form("simple_prediction_form"):
        st.header("Isi 5 Data Kunci")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0, step=0.5, key='simple_weight')
            height = st.number_input('Tinggi (meter)', 1.40, 2.20, 1.70, format="%.2f", key='simple_height')
        with col2:
            age = st.slider('Umur', 14, 65, 25, key='simple_age')
            gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'], key='simple_gender')
        family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ['yes', 'no'], key='simple_family')

        submitted = st.form_submit_button('**Prediksi Cepat**', use_container_width=True, type="primary")

        if submitted:
            if all(v is not None for v in [simple_model, le, simple_model_columns]):
                data = {'Weight': weight, 'Height': height, 'Age': age, 'Gender': gender, 'family_history_with_overweight': family_history}
                
                input_df = pd.DataFrame([data])
                processed_input = map_inputs(input_df)
                final_df = processed_input.reindex(columns=simple_model_columns, fill_value=0)
                
                prediction_encoded = simple_model.predict(final_df)
                prediction_text = le.inverse_transform(prediction_encoded)[0]
                
                st.divider()
                st.success(f"**Hasil Prediksi:** Anda masuk dalam kategori **{prediction_text.replace('_', ' ')}**")
                st.snow()
            else:
                st.error("Model atau aset lainnya gagal dimuat. Tidak bisa melakukan prediksi.")


def show_project_process_page():
    st.title("🔬 Alur Kerja Proyek: Dari Data Mentah ke Model")
    st.write("Bagian ini menceritakan langkah-langkah yang dilakukan dalam proyek ini.")

    tabs = st.tabs([
        "1. EDA", 
        "2. Preprocessing", 
        "3. Korelasi", 
        "4. Perbandingan Model", 
        "5. Optimasi Model"
    ])

    with tabs[0]:
        st.header("Analisis Data Eksplorasi (EDA)")
        st.write("Memahami distribusi dan karakteristik data awal.")
        try:
            st.image('distribusi_target.png', caption='Distribusi setiap kategori obesitas dalam dataset.')
            # Tambahkan visualisasi lain jika ada
        except FileNotFoundError:
            st.warning("Pastikan Anda sudah menyimpan gambar 'distribusi_target.png' dari notebook.")

    with tabs[1]:
        st.header("Pra-Pemrosesan Data")
        st.write("Membersihkan dan menyiapkan data sebelum pemodelan.")
        st.markdown("""
        Beberapa langkah kunci yang dilakukan:
        - **Menangani Nilai Hilang**: Mengisi data kosong dengan *median* (untuk angka) dan *modus* (untuk teks).
        - **Menghapus Data Duplikat**: Membersihkan baris data yang identik.
        - **Menangani Outlier**: Menyesuaikan nilai ekstrem menggunakan metode IQR.
        - **Encoding & Normalisasi**: Mengubah semua data menjadi format angka dan menyamakan skalanya.
        - **Penyeimbangan Kelas (SMOTE)**: Menyamakan jumlah data untuk setiap kategori target.
        """)
        
    with tabs[2]:
        st.header("Analisis Korelasi")
        st.write("Melihat hubungan statistik antar fitur.")
        try:
            st.image('heatmap_korelasi.png', caption='Heatmap menunjukkan korelasi antar fitur.')
        except FileNotFoundError:
            st.warning("Pastikan Anda sudah menyimpan gambar 'heatmap_korelasi.png' dari notebook.")

    with tabs[3]:
        st.header("Perbandingan Model Awal (Baseline)")
        st.write("Menguji beberapa algoritma dengan parameter default.")
        try:
            st.image('perbandingan_model_awal.png', caption='Perbandingan akurasi dari model-model awal.')
            df_hasil_awal = pd.read_csv('hasil_model_awal.csv')
            st.dataframe(df_hasil_awal)
        except FileNotFoundError:
            st.warning("Pastikan Anda sudah menyimpan 'perbandingan_model_awal.png' dan 'hasil_model_awal.csv' dari notebook.")
            
    with tabs[4]:
        st.header("Optimasi Model (Hyperparameter Tuning)")
        st.write("Mencari kombinasi parameter terbaik untuk meningkatkan performa model.")
        try:
            st.image('perbandingan_tuning.png', caption='Perbandingan akurasi sebelum dan sesudah hyperparameter tuning.')
            st.markdown("""
            Dari hasil optimasi, ditemukan bahwa **model XGBoost dengan parameter default** ternyata memberikan performa sedikit lebih baik daripada versi yang di-tuning. Ini menunjukkan parameter bawaan XGBoost sudah sangat kuat untuk dataset ini. Oleh karena itu, model default inilah yang dipilih sebagai model final.
            """)
        except FileNotFoundError:
            st.warning("Pastikan Anda sudah menyimpan gambar 'perbandingan_tuning.png' dari notebook.")


def show_model_performance_page():
    st.title("🏆 Performa Model Final (XGBoost)")
    st.write("Bagian ini menunjukkan seberapa baik performa model XGBoost terpilih, berdasarkan pengujian dengan data yang belum pernah dilihat sebelumnya.")

    try:
        st.subheader("1. Laporan Klasifikasi (Classification Report)")
        st.image('classification_report.png', caption='Hasil Uji Performa Model XGBoost Final.')
        st.divider()
        st.subheader("2. Pentingnya Fitur (Feature Importance)")
        st.image('feature_importance.png', caption='Peringkat Fitur berdasarkan Kontribusinya pada Model.')
    except FileNotFoundError:
        st.error("Pastikan file gambar 'classification_report.png' dan 'feature_importance.png' ada di repository Anda.")


# =====================================================================================
# --- NAVIGASI SIDEBAR DAN KONTROL HALAMAN ---
# =====================================================================================
with st.sidebar:
    # st.image("logo.png", use_column_width=True) # Ganti dengan nama file logo Anda jika ada
    st.title("Navigasi Dasbor")
    
    # Inisialisasi state halaman
    if 'page' not in st.session_state:
        st.session_state.page = 'Beranda'

    # Tombol navigasi
    if st.button("🏠 Beranda", use_container_width=True):
        st.session_state.page = 'Beranda'
    if st.button("👨‍⚕️ Prediksi Lengkap", use_container_width=True):
        st.session_state.page = 'Prediksi Lengkap'
    if st.button("⚡ Prediksi Ringkas", use_container_width=True):
        st.session_state.page = 'Prediksi Ringkas'
    if st.button("🔬 Proses Proyek", use_container_width=True):
        st.session_state.page = 'Proses Proyek'
    if st.button("🏆 Performa Model", use_container_width=True):
        st.session_state.page = 'Performa Model'

    st.divider()
    st.info("Dasbor ini dibuat untuk mendemonstrasikan alur kerja proyek Machine Learning.")


# Kontrol untuk menampilkan halaman yang sesuai
if st.session_state.page == 'Beranda':
    show_home_page()
elif st.session_state.page == 'Prediksi Lengkap':
    show_full_prediction_page()
elif st.session_state.page == 'Prediksi Ringkas':
    show_simple_prediction_page()
elif st.session_state.page == 'Proses Proyek':
    show_project_process_page()
elif st.session_state.page == 'Performa Model':
    show_model_performance_page()
