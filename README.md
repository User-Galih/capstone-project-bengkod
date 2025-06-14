# 💪 BodyFit Classifier - Capstone Project Bengkod

Selamat datang di repositori *Capstone Project Bengkod*!  
Aplikasi ini dikembangkan untuk memprediksi **kategori berat badan seseorang** berdasarkan input sederhana yang mudah diisi oleh pengguna.

---

## 📘 Project Background

Berdasarkan hasil seleksi fitur menggunakan metode **ANOVA F-score**, saya memilih 5 fitur teratas yang memiliki korelasi paling tinggi terhadap label klasifikasi, yaitu:

- **Weight** (Berat Badan)  
- **Gender** (Jenis Kelamin)  
- **family_history_with_overweight** (Riwayat Keluarga yang Kelebihan Berat Badan)  
- **FCVC** (Frekuensi Konsumsi Sayur)  
- **Age** (Usia)  

Namun, demi kemudahan dan relevansi dalam proses deployment, fitur **FCVC** digantikan oleh **Height** (Tinggi Badan).  
Keputusan ini mengacu pada pendekatan yang digunakan oleh **WHO** dan metode **Body Mass Index (BMI)**, di mana berat dan tinggi badan menjadi indikator utama untuk menilai status gizi seseorang.

---

## 🚀 Live Demo

🟢 Aplikasi ini dapat dicoba secara langsung melalui Streamlit:  
👉 [https://capstone-project-bengkod-galihputra-14359.streamlit.app/](https://capstone-project-bengkod-galihputra-14359.streamlit.app/)

---

## 📂 Struktur File Utama

```bash
├── Assets/
│   ├── model_xgb_tuned.pkl           # Model machine learning hasil pelatihan
│   ├── scaler.pkl                    # StandardScaler untuk data input
│   └── selected_features.pkl         # Fitur yang digunakan
├── Bengkod.py                        # File utama untuk Streamlit app
├── README.md                         # Penjelasan proyek


## 🧠 Teknologi yang Digunakan

- Python 3
- Streamlit (untuk UI interaktif)
- Scikit-learn (preprocessing dan evaluasi)
- XGBoost (algoritma klasifikasi)
- Joblib (penyimpanan model)
- Pandas dan NumPy (manipulasi data)

---

## 🔗 Referensi

- [BMI Calculator – Calculator.net](https://www.calculator.net/bmi-calculator.html)  
- [Truth About Weight – Global Initiative](https://www.truthaboutweight.global/)

---

## 👨‍💻 Developer

**Galih Putra Pratama**  
NIM: A11.2022.14359  
Universitas Dian Nuswantoro  
GitHub: [https://github.com/User-Galih](https://github.com/User-Galih)

---

## 📎 Repository & Deployment

- 🔗 **GitHub Repo**: [https://github.com/User-Galih/capstone-project-bengkod/](https://github.com/User-Galih/capstone-project-bengkod/)
- 🚀 **Live App**: [https://capstone-project-bengkod-galihputra-14359.streamlit.app/](https://capstone-project-bengkod-galihputra-14359.streamlit.app/)
