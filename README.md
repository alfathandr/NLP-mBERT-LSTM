# Analisis Sentimen Teks Mixed-Code Menggunakan mBERT-LSTM dan Teknik Balancing Data

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.10%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Repositori ini berisi implementasi model hybrid **mBERT-LSTM** untuk tugas analisis sentimen pada teks _code-mixed_ (campuran bahasa). Proyek ini juga melakukan studi komparatif terhadap beberapa teknik _data balancing_ untuk menangani masalah dataset yang tidak seimbang.

---

## 📖 Dataset

Proyek ini menggunakan dataset **DravidianCodeMix FIRE 2020**.

- **Sumber Data**: Komentar dari platform YouTube.
- **Karakteristik**:
  - **Bahasa**: Teks _code-mixed_ yang mengandung campuran Bahasa Tamil dan Inggris.
  - **Isi**: Teks bersifat informal, mengandung bahasa gaul, singkatan, dan potensi kesalahan ketik, yang mencerminkan bahasa percakapan alami di media sosial.
  - **Label Asli**: `Positive`, `Negative`, `Mixed_feelings`, `not-Tamil`, `unknown_state`.
- **Pra-pemrosesan pada Proyek Ini**:
  - Dataset difilter untuk hanya menggunakan 3 kelas sentimen utama: `Positive`, `Negative`, dan `Mixed_feelings`.
  - Label `Mixed_feelings` dipetakan menjadi `Neutral` untuk menyederhanakan tugas klasifikasi menjadi 3 kelas standar (Positif, Negatif, Netral).

---

## 🚀 Fitur Utama

- **Arsitektur Hibrida**: Menggabungkan kekuatan **mBERT** untuk pemahaman konteks multibahasa dengan **Bi-directional LSTM** untuk menangani dependensi sekuensial dalam teks.
- **Eksperimen Balancing Data**: Implementasi dan perbandingan tiga metode penyeimbangan data yang berbeda:
  - Random Over Sampling (**ROS**)
  - Random Under Sampling (**RUS**)
  - Hibrida (**ROS + ENN**)
- **Pipeline Terpadu**: Skrip yang terstruktur dan modular untuk setiap tahapan, mulai dari preparasi data, training, hingga evaluasi.
- **Antarmuka Baris Perintah (CLI)**: Semua skrip utama (`prepare`, `train`, `evaluate`) dijalankan melalui terminal dengan argumen yang jelas, mempermudah proses eksperimen.
- **Prediksi Interaktif**: Dilengkapi dengan skrip `predict.py` untuk berinteraksi langsung dengan model yang telah dilatih.

---

## 📁 Struktur Direktori

Berikut adalah tata letak utama dari proyek ini:

```bash
NLP-MBERT-LSTM/
├── dataset/
│ ├── kannada_sentiment_train.csv # Dataset mentah
│ ├── mal_full_sentiment_train.csv # Dataset mentah
│ ├── tamil_sentiment_full_train.csv # Dataset mentah, default digunakan.
│ └── ... (dataset hasil balancing)
│
├── saved_models/
│ ├── mbert_lstm_ros_3class.h5 # Bobot model
│ ├── tokenizer_ros_3class/ # Direktori tokenizer
│ ├── label_encoder_ros_3class.pkl # File label encoder
│ └── ... (artefak lain untuk RUS dan ROS+ENN)
│
├── 01_data_preparation.py # Skrip untuk preparasi & balancing data
├── 02_train.py # Skrip untuk training model
├── 03_evaluate.py # Skrip untuk evaluasi model
├── predict.py # Skrip untuk prediksi interaktif
├── README.md # Dokumentasi proyek ini
└── requirements.txt # Daftar pustaka Python yang dibutuhkan
```

---

## 🛠️ Instalasi

Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**

    ```bash
    git clone https://github.com/alfathandr/NLP-mBERT-LSTM
    cd NLP-mBERT-LSTM
    ```

2.  **Instal semua pustaka yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Seluruh pipeline dapat dijalankan melalui terminal. Pastikan Anda sudah mengunduh dataset mentah ke dalam folder `dataset`.

#### 1. Preparasi Data

Pilih salah satu metode (`ros`, `rus`, `ros_enn`) untuk membuat dataset seimbang.

# Contoh menggunakan metode ROS+ENN

python 01_data_preparation.py --method ros_enn

# Contoh training pada data hasil ROS+ENN

python 02_train.py --method ros_enn

# Contoh evaluasi model ROS+ENN

python 03_evaluate.py --method ros_enn

# Contoh menggunakan model ROS+ENN untuk prediksi

Gunakan model untuk memprediksi sentimen dari kalimat baru.
python predict.py --method ros_enn
