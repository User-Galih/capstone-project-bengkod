import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder # Ensure LabelEncoder is imported here too
from sklearn.preprocessing import StandardScaler # Also import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dasbor Proyek Obesitas",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DEFINISIKAN PATH ASET ---
ASSETS_PATH = 'Assets' # Pastikan ini adalah lokasi folder Assets Anda

# --- FUNGSI-FUNGSI PEMUAT SUMBER DAYA ---
@st.cache_resource
def load_full_model_resources():
    """Memuat model LENGKAP dan aset-asetnya."""
    try:
        model_path = os.path.join(ASSETS_PATH, 'BengKod_Default_XGBoost_Model.pkl')
        le_target_path = os.path.join(ASSETS_PATH, 'label_encoder.pkl') # Ini untuk target
        model_cols_path = os.path.join(ASSETS_PATH, 'model_columns.pkl')
        scaler_path = os.path.join(ASSETS_PATH, 'scaler.pkl') # Path to the scaler

        model = joblib.load(model_path)
        le_target = joblib.load(le_target_path)
        model_cols = joblib.load(model_cols_path)
        scaler = joblib.load(scaler_path) # Load the scaler

        # --- MEMUAT LABEL ENCODER UNTUK FITUR KATEGORIKAL ---
        feature_encoders = {}
        categorical_features = [ # List ini harus sama persis dengan yang di notebook
            'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS'
        ]
        for col in categorical_features:
            encoder_file = os.path.join(ASSETS_PATH, f'le_feature_{col}.pkl') # Name of the saved encoder file
            feature_encoders[col] = joblib.load(encoder_file)
        
        return model, le_target, model_cols, scaler, feature_encoders # Return the scaler and feature encoders
    except Exception as e:
        st.error(f"Gagal memuat aset model lengkap dari folder '{ASSETS_PATH}': {e}")
        st.error(f"Detail kesalahan: {e}")
        return None, None, None, None, None # Add None for scaler and feature encoders

@st.cache_resource
def load_simple_model_resources():
    """Memuat model RINGKAS dan aset-asetnya."""
    try:
        simple_model_path = os.path.join(ASSETS_PATH, 'Model_Ringkas_XGBoost.pkl')
        le_simple_path = os.path.join(ASSETS_PATH, 'label_encoder.pkl') # Assuming same for simple model
        model_cols_simple_path = os.path.join(ASSETS_PATH, 'model_columns_ringkas.pkl')
        scaler_path = os.path.join(ASSETS_PATH, 'scaler.pkl') # Assuming same scaler
        
        model_simple = joblib.load(simple_model_path)
        le_simple = joblib.load(le_simple_path)
        model_cols_simple = joblib.load(model_cols_simple_path)
        scaler_simple = joblib.load(scaler_path) # Load the scaler for simple model

        # If simple model also uses LabelEncoder for features, load them here too
        # Based on your notebook snippet (BMI section), 'Gender' and 'family_history_with_overweight'
        # are also LabelEncoded in the simple model context.
        feature_encoders_simple = {}
        categorical_features_simple = ['Gender', 'family_history_with_overweight'] # Features used by simple model
        for col in categorical_features_simple:
            encoder_file = os.path.join(ASSETS_PATH, f'le_feature_{col}.pkl')
            feature_encoders_simple[col] = joblib.load(encoder_file)

        return model_simple, le_simple, model_cols_simple, scaler_simple, feature_encoders_simple
    except Exception as e:
        st.error(f"Gagal memuat aset model ringkas dari folder '{ASSETS_PATH}': {e}")
        return None, None, None, None, None

# --- MEMUAT SEMUA SUMBER DAYA ---
# Ensure correct unpacking based on what load_full_model_resources returns
full_model, le, full_model_columns, scaler, feature_encoders = load_full_model_resources()
simple_model, le_simple, simple_model_columns, scaler_simple, feature_encoders_simple = load_simple_model_resources()

# --- FUNGSI PEMETAAN INPUT (DENGAN ENCODING YANG SUDAH DIPERBAIKI) ---
# This function is used to apply Label Encoding to categorical features
def map_inputs(input_df, encoders):
    """Mengubah input teks dari pengguna menjadi angka sesuai training (urutan alfabetis)
    menggunakan LabelEncoders yang sudah dilatih."""
    df = input_df.copy()
    
    # Iterate through the encoders provided
    for col, le_obj in encoders.items():
        if col in df.columns:
            try:
                df[col] = le_obj.transform(df[col])
            except ValueError as e:
                # Handle cases where an unknown category is encountered
                st.warning(f"Kategori tidak dikenal untuk kolom '{col}': {df[col].iloc[0]}. "
                           "Ini mungkin memengaruhi akurasi prediksi. "
                           "Pastikan semua kategori yang mungkin sudah ada dalam data pelatihan.")
                # You might choose a different strategy for unknown categories,
                # e.g., setting to 0 or a mode value if appropriate.
                df[col] = -1 # Or a default numeric value if your model can handle it
        else:
            st.warning(f"Kolom '{col}' tidak ditemukan di input data. "
                       "Ini mungkin terjadi jika model dilatih dengan fitur ini tetapi tidak ada di input.")
            # Depending on your model, you might need to add this column with a default value (e.g., 0)
            df[col] = 0 # Example default value

    return df

# =====================================================================================
# --- FUNGSI UNTUK HALAMAN-HALAMAN APLIKASI ---
# =====================================================================================

def show_home_page():
    st.title("🚀 Selamat Datang di Dasbor Proyek Obesitas")
    st.markdown("""
    Aplikasi ini adalah demonstrasi lengkap dari sebuah proyek machine learning, mulai dari analisis data hingga deployment model prediktif.
    Gunakan menu navigasi di sebelah kiri untuk menjelajahi setiap bagian dari dasbor ini.
    """)
    try:
        st.image(os.path.join(ASSETS_PATH, "logo.png"), width=400)
    except Exception:
        st.info("Anda bisa menambahkan gambar 'logo.png' ke dalam folder 'Assets' Anda.")

def show_full_prediction_page():
    st.title("👨‍⚕️ Prediksi Lengkap (Model XGBoost)")
    st.info("Gunakan semua fitur gaya hidup untuk mendapatkan prediksi yang paling komprehensif dari model Machine Learning.", icon="💡")

    # Panggil fungsi pemuatan aset
    # Ensure correct unpacking
    model, le_target, model_columns, scaler, feature_encoders = load_full_model_resources()

    if all(v is None for v in [model, le_target, model_columns, scaler, feature_encoders]):
        st.warning("Aset model tidak berhasil dimuat. Prediksi tidak dapat dilakukan.")
        return # Hentikan eksekusi jika aset tidak ada

    st.subheader("Isi Data Diri Anda")

    # --- FORM INPUT DENGAN DESAIN BARU ---
    with st.expander("🧍‍♂️ **Data Diri & Fisik**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'], key='full_gender')
            age = st.slider('Umur', 14, 65, 25, key='full_age')
        with col2:
            height = st.number_input('Tinggi (meter)', 1.40, 2.20, 1.70, format="%.2f", key='full_height')
            weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0, step=0.5, key='full_weight')
        
        family_history = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ['no', 'yes'], key='full_family')

    with st.expander("🍎 **Pola Makan & Minum**"):
        col3, col4 = st.columns(2)
        with col3:
            favc = st.selectbox('Sering konsumsi makanan tinggi kalori?', ['yes', 'no'], key='full_favc')
            ncp = st.slider('Jumlah makan utama per hari', 1.0, 4.0, 3.0, step=0.1, key='full_ncp') # Mengubah ke float
            scc = st.selectbox('Monitor kalori makanan?', ['no', 'yes'], key='full_scc')
        with col4:
            fcvc = st.slider('Frekuensi konsumsi sayuran (1-3)', 1.0, 3.0, 2.0, step=0.1, key='full_fcvc') # Mengubah ke float
            ch2o = st.slider('Konsumsi air harian (1-3 liter)', 1.0, 3.0, 2.0, step=0.1, key='full_ch2o') # Mengubah ke float
        
        caec = st.selectbox('Konsumsi makanan di antara waktu makan', ['Sometimes', 'Frequently', 'Always', 'no'], key='full_caec')

    with st.expander("🏃‍♂️ **Aktivitas & Gaya Hidup**"):
        col5, col6 = st.columns(2)
        with col5:
            smoke = st.selectbox('Apakah Anda merokok?', ['no', 'yes'], key='full_smoke')
            faf = st.slider('Frekuensi aktivitas fisik (0-3 hari/minggu)', 0.0, 3.0, 1.0, step=0.1, key='full_faf') # Mengubah ke float
        with col6:
            tue = st.slider('Waktu penggunaan gadget (0-2 jam/hari)', 0.0, 2.0, 1.0, step=0.1, key='full_tue') # Mengubah ke float
            calc = st.selectbox('Frekuensi konsumsi alkohol', ['no', 'Sometimes', 'Frequently', 'Always'], key='full_calc')
        
        mtrans = st.selectbox('Transportasi Utama', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'], key='full_mtrans')


    if st.button('**Prediksi Sekarang**', use_container_width=True, type="primary", key='predict_full'):
        # Memastikan model dan encoder berhasil dimuat
        if model and le_target and model_columns and scaler and feature_encoders:
            # Mengumpulkan data input
            input_data = {
                'Gender': gender,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'family_history_with_overweight': family_history,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }

            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])

            # --- Apply Label Encoding for categorical features ---
            # Use the map_inputs function with the loaded feature_encoders
            input_df_encoded = map_inputs(input_df, feature_encoders)

            # --- Identify numerical and categorical columns for scaling ---
            numerical_cols_for_scaling = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            categorical_cols_already_encoded = [col for col in feature_encoders.keys() if col in input_df_encoded.columns] # These are already numerical due to LE

            # Separate numerical features for scaling
            numerical_data_for_scaling = input_df_encoded[numerical_cols_for_scaling]
            # Separate categorical features (already encoded)
            categorical_data_encoded = input_df_encoded[categorical_cols_already_encoded]

            # --- Apply StandardScaler to numerical features ---
            # Ensure the order of numerical columns matches the order used during scaler.fit_transform
            # This is implicitly handled if numerical_cols_for_scaling list is consistent.
            scaled_numerical_data = scaler.transform(numerical_data_for_scaling)
            scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_cols_for_scaling, index=input_df.index)

            # --- Recombine scaled numerical and encoded categorical features ---
            # Ensure all columns expected by model_columns are present and in the correct order
            # This might require creating a dummy DataFrame with all model_columns
            # and then filling it.
            
            # Create a DataFrame with all columns the model expects, initialized to 0
            final_input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

            # Fill in the scaled numerical data
            for col in numerical_cols_for_scaling:
                final_input_df[col] = scaled_numerical_df[col]

            # Fill in the encoded categorical data
            for col in categorical_cols_already_encoded:
                final_input_df[col] = categorical_data_encoded[col]

            # Ensure final_input_df has exactly the columns and order as model_columns
            # This .reindex operation is crucial to align columns
            # final_df = final_input_df.reindex(columns=model_columns, fill_value=0) # This might be redundant if the above is precise

            # Convert to numpy array for prediction
            final_numpy = final_input_df.to_numpy()

            # DEBUGGING: Print final input to model
            # st.write("Final Input to Model (after all preprocessing and scaling):")
            # st.write(final_input_df)
            # st.write("Columns of final input:")
            # st.write(final_input_df.columns.tolist())


            # Lakukan Prediksi
            prediction_encoded = model.predict(final_numpy)
            prediction_text = le_target.inverse_transform(prediction_encoded)[0]

            # Tampilkan Hasil
            st.divider()
            st.success(f"**Hasil Prediksi:** Anda masuk dalam kategori **{prediction_text.replace('_', ' ')}**")
            st.balloons()
        else:
            st.error("Terjadi masalah saat memuat model atau aset. Prediksi tidak dapat dilakukan.")

def show_simple_prediction_page():
    st.title("⚡ Prediksi Ringkas (5 Fitur Utama)")
    st.info("💡 Dapatkan prediksi cepat hanya dengan 5 fitur kunci yang paling berpengaruh.", icon="💡")

    # Panggil fungsi pemuatan aset
    model_simple, le_simple, simple_model_columns, scaler_simple, feature_encoders_simple = load_simple_model_resources()

    if all(v is None for v in [model_simple, le_simple, simple_model_columns, scaler_simple, feature_encoders_simple]):
        st.warning("Aset model ringkas tidak berhasil dimuat. Prediksi tidak dapat dilakukan.")
        return

    st.subheader("Isi 5 Data Kunci")

    col1, col2 = st.columns(2)
    with col1:
        weight_s = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0, step=0.5, key='simple_weight')
        height_s = st.number_input('Tinggi (meter)', 1.40, 2.20, 1.70, format="%.2f", key='simple_height')
    with col2:
        age_s = st.slider('Umur', 14, 65, 25, key='simple_age')
        gender_s = st.selectbox('Jenis Kelamin', ['Male', 'Female'], key='simple_gender')
    family_history_s = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', ['no', 'yes'], key='simple_family')

    if st.button('**Prediksi Cepat**', use_container_width=True, type="primary", key='predict_simple'):
        if model_simple and le_simple and simple_model_columns and scaler_simple and feature_encoders_simple:
            input_data_s = {
                'Weight': weight_s,
                'Height': height_s,
                'Age': age_s,
                'Gender': gender_s,
                'family_history_with_overweight': family_history_s
            }
            input_df_s = pd.DataFrame([input_data_s])

            # Apply Label Encoding for categorical features
            input_df_encoded_s = map_inputs(input_df_s, feature_encoders_simple)

            # Identify numerical and categorical columns for the simple model
            numerical_cols_simple = ['Weight', 'Height', 'Age']
            categorical_cols_simple_encoded = [col for col in feature_encoders_simple.keys() if col in input_df_encoded_s.columns]

            # Separate numerical features for scaling
            numerical_data_for_scaling_s = input_df_encoded_s[numerical_cols_simple]
            categorical_data_encoded_s = input_df_encoded_s[categorical_cols_simple_encoded]

            # Apply StandardScaler to numerical features for the simple model
            scaled_numerical_data_s = scaler_simple.transform(numerical_data_for_scaling_s)
            scaled_numerical_df_s = pd.DataFrame(scaled_numerical_data_s, columns=numerical_cols_simple, index=input_df_s.index)

            # Recombine scaled numerical and encoded categorical features
            final_input_df_s = pd.DataFrame(0.0, index=[0], columns=simple_model_columns)
            for col in numerical_cols_simple:
                final_input_df_s[col] = scaled_numerical_df_s[col]
            for col in categorical_cols_simple_encoded:
                final_input_df_s[col] = categorical_data_encoded_s[col]
            
            final_numpy_s = final_input_df_s.to_numpy()

            prediction_encoded_s = model_simple.predict(final_numpy_s)
            prediction_text_s = le_simple.inverse_transform(prediction_encoded_s)[0]

            st.divider()
            st.success(f"**Hasil Prediksi:** Anda masuk dalam kategori **{prediction_text_s.replace('_', ' ')}**")
            st.snow()
        else:
            st.error("Terjadi masalah saat memuat model ringkas atau aset. Prediksi tidak dapat dilakukan.")

def show_project_process_page():
    st.title("🔬 Alur Kerja Proyek: Dari Data Mentah ke Model")
    st.write("Bagian ini menceritakan langkah-langkah yang dilakukan dalam proyek ini.")

    tabs = st.tabs([
        "1. EDA", "2. Preprocessing", "3. Korelasi", 
        "4. Perbandingan Model", "5. Optimasi Model"
    ])

    with tabs[0]:
        st.header("Analisis Data Eksplorasi (EDA)")
        try:
            st.image(os.path.join(ASSETS_PATH, 'distribusi_target.png'), caption='Distribusi setiap kategori obesitas dalam dataset.')
            st.image(os.path.join(ASSETS_PATH, 'distribusi_numerik.png'), caption='Distribusi fitur-fitur numerik.')
            st.image(os.path.join(ASSETS_PATH, 'boxplot_outlier.png'), caption='Boxplot untuk mendeteksi outlier.')
        except FileNotFoundError as e:
            st.warning(f"Satu atau lebih file gambar EDA tidak ditemukan di folder Assets. Error: {e}")

    with tabs[1]:
        st.header("Pra-Pemrosesan Data")
        st.markdown("""
        Langkah-langkah kunci yang dilakukan:
        - **Menangani Nilai Hilang**: Mengisi data kosong dengan *median* dan *modus*.
        - **Menghapus Data Duplikat**: Membersihkan baris data yang identik.
        - **Menangani Outlier**: Menyesuaikan nilai ekstrem menggunakan metode IQR.
        - **Encoding Fitur**: Mengubah semua data teks menjadi format angka.
        - **Penyeimbangan Kelas (SMOTE)**: Menyamakan jumlah data untuk setiap kategori target.
        - **Normalisasi Data**: Menyamakan skala semua fitur numerik ke rentang 0-1.
        """)
        
    with tabs[2]:
        st.header("Analisis Korelasi")
        try:
            st.image(os.path.join(ASSETS_PATH, 'heatmap_korelasi.png'), caption='Heatmap menunjukkan korelasi antar fitur.')
        except FileNotFoundError:
            st.warning("File 'heatmap_korelasi.png' tidak ditemukan di folder Assets.")

    with tabs[3]:
        st.header("Perbandingan Model Awal (Baseline)")
        try:
            st.image(os.path.join(ASSETS_PATH, 'perbandingan_model_awal.png'), caption='Perbandingan akurasi dari model-model awal.')
            df_hasil_awal = pd.read_csv(os.path.join(ASSETS_PATH, 'hasil_model_awal.csv'))
            st.dataframe(df_hasil_awal)
        except FileNotFoundError:
            st.warning("Pastikan 'perbandingan_model_awal.png' dan 'hasil_model_awal.csv' ada di folder Assets.")
            
    with tabs[4]:
        st.header("Optimasi Model (Hyperparameter Tuning)")
        try:
            st.image(os.path.join(ASSETS_PATH, 'perbandingan_tuning.png'), caption='Perbandingan akurasi sebelum dan sesudah hyperparameter tuning.')
            st.markdown("""
            Dari hasil optimasi, ditemukan bahwa **model XGBoost dengan parameter default** ternyata memberikan performa sedikit lebih baik daripada versi yang di-tuning. Oleh karena itu, model default inilah yang dipilih sebagai model final.
            """)
        except FileNotFoundError:
            st.warning("File 'perbandingan_tuning.png' tidak ditemukan di folder Assets.")


def show_model_performance_page():
    st.title("🏆 Performa Model Final (XGBoost)")
    st.write("Bagian ini menunjukkan seberapa baik performa model XGBoost terpilih.")

    try:
        st.subheader("1. Laporan Klasifikasi")
        st.image(os.path.join(ASSETS_PATH, 'classification_report.png'), caption='Hasil Uji Performa Model XGBoost Final.')
        st.divider()
        st.subheader("2. Pentingnya Fitur (Feature Importance)")
        st.image(os.path.join(ASSETS_PATH, 'feature_importance.png'), caption='Peringkat Fitur berdasarkan Kontribusinya pada Model.')
    except FileNotFoundError:
        st.error(f"Pastikan 'classification_report.png' dan 'feature_importance.png' ada di folder Assets Anda.")


# =====================================================================================
# --- NAVIGASI SIDEBAR DAN KONTROL HALAMAN ---
# =====================================================================================
with st.sidebar:
    try:
        st.image(os.path.join(ASSETS_PATH, "logo.png"), use_container_width=True)
    except Exception:
        pass 
    
    st.title("Navigasi Dasbor")
    
    if 'page' not in st.session_state:
        st.session_state.page = 'Beranda'

    pages = {
        "Beranda": "🏠 Beranda",
        "Prediksi Lengkap": "👨‍⚕️ Prediksi Lengkap",
        "Prediksi Ringkas": "⚡ Prediksi Ringkas",
        "Proses Proyek": "🔬 Proses Proyek",
        "Performa Model": "🏆 Performa Model"
    }
    
    for page_key, page_title in pages.items():
        if st.button(page_title, use_container_width=True, type='primary' if st.session_state.page == page_key else 'secondary'):
            st.session_state.page = page_key
    
    st.divider()
    st.info("Dasbor ini dibuat untuk mendemonstrasikan alur kerja proyek Machine Learning.")


# Kontrol untuk menampilkan halaman yang sesuai
page_functions = {
    'Beranda': show_home_page,
    'Prediksi Lengkap': show_full_prediction_page,
    'Prediksi Ringkas': show_simple_prediction_page,
    'Proses Proyek': show_project_process_page,
    'Performa Model': show_model_performance_page
}

if st.session_state.page in page_functions:
    page_functions[st.session_state.page]()
else:
    show_home_page()
