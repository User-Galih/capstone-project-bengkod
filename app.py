import streamlit as st
import pandas as pd
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dasbor Proyek Obesitas",
    page_icon="🚀",
    layout="wide"
)

ASSETS_PATH = "Assets"

@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load(os.path.join(ASSETS_PATH, 'BengKod_Default_XGBoost_Model.pkl'))
        columns = joblib.load(os.path.join(ASSETS_PATH, 'model_columns.pkl'))
        label_encoder = joblib.load(os.path.join(ASSETS_PATH, 'label_encoder.pkl'))
        scaler = joblib.load(os.path.join(ASSETS_PATH, 'minmax_scaler.pkl'))  # ⬅️ Tambahan
        return model, columns, label_encoder, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau file lainnya: {e}")
        return None, None, None, None

model, model_columns, label_encoder, scaler = load_model_resources()

# Fitur numerik yang dinormalisasi saat training
numerical_features = ['Age', 'Weight', 'Height', 'FAF', 'NCP', 'CH2O', 'FCVC', 'TUE']

def map_inputs(df):
    df = df.copy()
    gender_map = {'Female': 0, 'Male': 1}
    yes_no_map = {'no': 0, 'yes': 1}
    caec_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
    calc_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
    mtrans_map = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}

    df['Gender'] = df['Gender'].map(gender_map)
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    df['FAVC'] = df['FAVC'].map(yes_no_map)
    df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    df['SCC'] = df['SCC'].map(yes_no_map)
    df['CAEC'] = df['CAEC'].map(caec_map)
    df['CALC'] = df['CALC'].map(calc_map)
    df['MTRANS'] = df['MTRANS'].map(mtrans_map)

    # 🧠 Normalisasi fitur numerik dengan scaler hasil training
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df

def predict(data):
    df = pd.DataFrame([data])
    df = map_inputs(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df)
    return label_encoder.inverse_transform(prediction)[0]

# --- BERANDA ---
def show_home():
    st.title("🚀 Selamat Datang di Dasbor Proyek Obesitas")
    st.write("Aplikasi ini mendemonstrasikan prediksi kategori obesitas berdasarkan data gaya hidup.")
    try:
        st.image(os.path.join(ASSETS_PATH, "logo.png"), width=400)
    except:
        pass

# --- PREDIKSI LENGKAP ---
def show_prediction():
    st.title("👨‍⚕️ Prediksi Kategori Obesitas Lengkap")
    with st.form("prediction_form"):
        st.subheader("Silakan isi data berikut:")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
            age = st.slider('Umur', 14, 65, 25)
            height = st.number_input('Tinggi (meter)', 1.4, 2.2, 1.7, format="%.2f")
            weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, 70.0)
            family_history = st.selectbox('Riwayat keluarga overweight?', ['yes', 'no'])
            favc = st.selectbox('Sering konsumsi makanan tinggi kalori?', ['yes', 'no'])
            fcvc = st.slider('Frekuensi konsumsi sayur (1-3)', 1.0, 3.0, 2.0)
            ncp = st.slider('Jumlah makan utama per hari', 1.0, 4.0, 3.0)

        with col2:
            caec = st.selectbox('Camilan di antara waktu makan?', ['no', 'Sometimes', 'Frequently', 'Always'])
            smoke = st.selectbox('Merokok?', ['yes', 'no'])
            ch2o = st.slider('Konsumsi air harian (1-3 liter)', 1.0, 3.0, 2.0)
            scc = st.selectbox('Kontrol kalori makanan?', ['yes', 'no'])
            faf = st.slider('Frekuensi aktivitas fisik', 0.0, 3.0, 1.0)
            tue = st.slider('Waktu pakai gadget (jam)', 0.0, 2.0, 1.0)
            calc = st.selectbox('Konsumsi alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
            mtrans = st.selectbox('Transportasi utama', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

        submitted = st.form_submit_button("Prediksi")

        if submitted:
            if model is None or model_columns is None or label_encoder is None or scaler is None:
                st.error("Model belum dimuat. Coba ulangi.")
            else:
                input_data = {
                    'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
                    'family_history_with_overweight': family_history,
                    'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec,
                    'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc, 'FAF': faf,
                    'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
                }
                result = predict(input_data)
                st.success(f"Prediksi kategori obesitas: {result.replace('_', ' ')}")

# --- MAIN ---
page = st.sidebar.selectbox("Navigasi", ["Beranda", "Prediksi Lengkap"])

if page == "Beranda":
    show_home()
elif page == "Prediksi Lengkap":
    show_prediction()
