# -*- coding: utf-8 -*-
"""
==========================================================================
01_DATA_PREPARATION.PY
==========================================================================
Author: Muh. Al Fathan
Project: Analisis Sentimen Code-Mixed mBERT-LSTM

Deskripsi:
-----------
Skrip ini adalah titik awal untuk semua eksperimen preparasi data.
Tujuannya adalah untuk mengambil dataset mentah dan menghasilkan versi
yang seimbang (balanced) menggunakan salah satu dari tiga metode yang
disediakan.

Metode Balancing yang Tersedia:
1.  **ROS (Random Over Sampling)**: Memperbanyak kelas minoritas.
2.  **RUS (Random Under Sampling)**: Mengurangi kelas mayoritas.
3.  **ROS+ENN (Hybrid)**: Kombinasi oversampling lalu membersihkan noise.

Input:
-------
- `dataset/tamil_sentiment_full_train.csv`: File CSV mentah.

Output:
--------
- File CSV yang sudah seimbang, dengan nama yang sesuai metode, contoh:
  - `dataset/train_ros_balanced_3class.csv`
  - `dataset/train_rus_balanced_3class.csv`
  - `dataset/train_ros_enn_balanced_3class.csv`

Cara Menjalankan (dari Terminal):
----------------------------------
Pilih metode yang ingin dijalankan dengan argumen `--method`.

# Untuk menjalankan metode ROS
$ python 01_data_preparation.py --method ros

# Untuk menjalankan metode RUS
$ python 01_data_preparation.py --method rus

# Untuk menjalankan metode ROS+ENN
$ python 01_data_preparation.py --method ros_enn
==========================================================================
"""

import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours

def load_and_clean_data(input_path):
    """
    Memuat dataset mentah dari path, membersihkan, dan memfilternya.
    Fungsi ini adalah langkah umum yang digunakan oleh semua metode balancing.
    """
    print(f"📖 Membaca dataset dari: {input_path}")
    try:
        # Dataset ini menggunakan tab sebagai pemisah (sep='\t').
        df = pd.read_csv(
            input_path, sep='\t', header=None,
            names=['text', 'label'], on_bad_lines='warn', quoting=3
        )
        print("✅ Dataset asli berhasil dimuat.")
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di '{input_path}'.")
        return None

    df.dropna(subset=['text', 'label'], inplace=True)
    
    print("🔍 Memfilter data untuk 3 kelas (Positive, Negative, Neutral)...")
    selected_labels = ['Positive', 'Negative', 'Mixed_feelings']
    df = df[df['label'].isin(selected_labels)].copy()
    df['label'] = df['label'].replace({'Mixed_feelings': 'Neutral'})
    
    print("\n" + "="*50)
    print("📊 DISTRIBUSI KELAS SEBELUM BALANCING")
    print("="*50)
    print(df['label'].value_counts())
    
    return df

def apply_ros(df):
    """Menerapkan Random Over Sampling (ROS) pada dataframe."""
    print("\n⚙️  Menerapkan Random Over Sampling (ROS)...")
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(df[['text']], df['label'])
    df_balanced = pd.DataFrame(X_res, columns=['text'])
    df_balanced['label'] = y_res
    return df_balanced

def apply_rus(df):
    """Menerapkan Random Under Sampling (RUS) pada dataframe."""
    print("\n⚙️  Menerapkan Random Under Sampling (RUS)...")
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(df[['text']], df['label'])
    df_balanced = pd.DataFrame(X_res, columns=['text'])
    df_balanced['label'] = y_res
    return df_balanced

def apply_ros_enn(df):
    """Menerapkan metode hibrida ROS + ENN pada dataframe."""
    # 1. Oversampling dengan ROS
    print("\n⚙️  LANGKAH 1: Menerapkan Random Over Sampling (ROS)...")
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(df[['text']], df['label'])
    df_ros = pd.DataFrame(X_ros, columns=['text'])
    df_ros['label'] = y_ros
    print(f"   Ukuran data setelah ROS: {df_ros.shape}")

    # 2. Cleaning dengan ENN
    print("\n⚙️  LANGKAH 2: Membersihkan data hasil oversampling dengan ENN...")
    print("   a. Mengubah teks menjadi vektor TF-IDF (temporer)...")
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df_ros['text'])
    
    print("   b. Mengidentifikasi dan menghapus sampel yang 'noisy'...")
    enn = EditedNearestNeighbours(n_jobs=-1)
    _, _ = enn.fit_resample(X_tfidf, df_ros['label'])
    
    cleaned_indices = enn.sample_indices_
    df_final = df_ros.iloc[cleaned_indices].reset_index(drop=True)
    return df_final

def main(method):
    """Fungsi utama yang mengorkestrasi seluruh proses."""
    
    # --- Konfigurasi Path ---
    DATASET_DIR = 'dataset'
    TRAIN_FILE = 'tamil_sentiment_full_train.csv'
    input_path = os.path.join(DATASET_DIR, TRAIN_FILE)
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # --- Muat dan Bersihkan Data ---
    df_cleaned = load_and_clean_data(input_path)
    if df_cleaned is None:
        return

    # --- Terapkan Metode Balancing yang Dipilih ---
    df_balanced = None
    if method == 'ros':
        df_balanced = apply_ros(df_cleaned)
    elif method == 'rus':
        df_balanced = apply_rus(df_cleaned)
    elif method == 'ros_enn':
        df_balanced = apply_ros_enn(df_cleaned)

    if df_balanced is not None:
        print("\n✅ Proses balancing selesai.")
        print(f"   Ukuran data akhir: {df_balanced.shape}")
        
        print("\n" + "="*50)
        print(f"📊 DISTRIBUSI KELAS AKHIR (SETELAH {method.upper()})")
        print("="*50)
        print(df_balanced['label'].value_counts())

        # --- Simpan Hasil ---
        # Nama file output dibuat dinamis berdasarkan metode yang dipilih.
        output_filename = f'train_{method}_balanced_3class.csv'
        output_path = os.path.join(DATASET_DIR, output_filename)
        
        print(f"\n💾 Menyimpan data yang sudah seimbang ke: {output_path}")
        df_balanced.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n🎉 Data latih seimbang ({method.upper()}) siap digunakan.")
    else:
        print(f"❌ ERROR: Metode '{method}' tidak dikenal.")

if __name__ == "__main__":
    # Bagian ini mengatur bagaimana skrip menerima argumen dari baris perintah.
    parser = argparse.ArgumentParser(description="Jalankan proses preparasi dan balancing data.")
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['ros', 'rus', 'ros_enn'],
        help="Metode balancing yang akan digunakan: 'ros', 'rus', atau 'ros_enn'."
    )
    args = parser.parse_args()
    
    # Menjalankan fungsi utama dengan metode yang dipilih.
    main(args.method)