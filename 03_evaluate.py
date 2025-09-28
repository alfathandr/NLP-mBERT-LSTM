# -*- coding: utf-8 -*-
"""
==========================================================================
03_EVALUATE.PY
==========================================================================
Author: Muh. Al Fathan
Project: Analisis Sentimen Code-Mixed mBERT-LSTM

Deskripsi:
-----------
Skrip ini bertujuan untuk mengevaluasi performa model mBERT-LSTM yang telah
dilatih oleh skrip `02_train.py`. Skrip ini akan memuat model dan artefak
yang sesuai berdasarkan metode balancing yang dipilih, lalu menghasilkan
laporan performa pada set data validasi yang relevan.

Proses yang dilakukan:
1.  Memuat artefak yang sesuai (tokenizer, label encoder, bobot model).
2.  Memuat kembali dataset dan mereplikasi pembagian data validasi.
3.  Membangun ulang arsitektur model yang identik.
4.  Melakukan prediksi pada data validasi.
5.  Menampilkan metrik performa (Laporan Klasifikasi & Confusion Matrix).

Input:
-------
- Artefak dari `02_train.py` (bobot, tokenizer, encoder)
- File CSV seimbang dari `01_data_preparation.py`

Output:
--------
- Tampilan Laporan Klasifikasi dan plot Confusion Matrix.

Cara Menjalankan (dari Terminal):
----------------------------------
# Untuk mengevaluasi model hasil training ROS
$ python 03_evaluate.py --method ros

# Untuk mengevaluasi model hasil training RUS
$ python 03_evaluate.py --method rus

# Untuk mengevaluasi model hasil training ROS+ENN
$ python 03_evaluate.py --method ros_enn
==========================================================================
"""

import os
import pickle
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model

# --- 1. Konfigurasi ---
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128

# Konfigurasi Path
DATASET_DIR = 'dataset'
SAVED_MODELS_DIR = 'saved_models'


def load_artifacts(method):
    """
    Memuat tokenizer dan label encoder yang sesuai dengan metode.
    """
    print(f"📖 Memuat tokenizer dan label encoder untuk metode '{method.upper()}'...")
    try:
        tokenizer_dirname = f'tokenizer_{method}_3class'
        encoder_filename = f'label_encoder_{method}_3class.pkl'
        
        tokenizer_path = os.path.join(SAVED_MODELS_DIR, tokenizer_dirname)
        label_encoder_path = os.path.join(SAVED_MODELS_DIR, encoder_filename)
        
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        print("✅ Artefak berhasil dimuat.")
        return tokenizer, label_encoder
    except (OSError, FileNotFoundError) as e:
        print(f"❌ ERROR: Gagal memuat artefak. Pastikan skrip training untuk metode '{method}' sudah dijalankan.")
        print(f"   Detail: {e}")
        return None, None


def load_validation_data(method):
    """
    Memuat dataset dan mengambil set validasi yang sama seperti saat training.
    """
    print("📖 Memuat data validasi...")
    dataset_filename = f'train_{method}_balanced_3class.csv'
    dataset_path = os.path.join(DATASET_DIR, dataset_filename)
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File dataset tidak ditemukan di '{dataset_path}'.")
        return None

    _, df_val = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df['label']
    )
    print("✅ Data validasi berhasil dipisahkan.")
    return df_val


def build_model(num_classes):
    """
    Membangun ulang arsitektur model yang identik dengan saat training.
    """
    class BertLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(BertLayer, self).__init__(**kwargs)
            self.bert = TFBertModel.from_pretrained(MODEL_NAME)
            self.bert.trainable = False # Wajib False untuk evaluasi
        
        def call(self, inputs):
            input_ids, attention_mask = inputs
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    input_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
    bert_output = BertLayer()([input_ids, attention_mask])
    bilstm = Bidirectional(LSTM(64, return_sequences=False))(bert_output)
    dropout = Dropout(0.3)(bilstm)
    dense_output = Dense(num_classes, activation='softmax', name='classifier')(dropout)
    
    model = Model(inputs=[input_ids, attention_mask], outputs=dense_output)
    return model


def main(method):
    """Fungsi utama untuk menjalankan pipeline evaluasi."""
    
    print(f"\n===== MEMULAI PROSES EVALUASI UNTUK METODE: {method.upper()} =====")
    
    # --- Tahap A: Memuat semua yang diperlukan ---
    tokenizer, label_encoder = load_artifacts(method)
    if tokenizer is None:
        return
        
    df_val = load_validation_data(method)
    if df_val is None:
        return

    num_classes = len(label_encoder.classes_)
    
    # --- Tahap B: Membangun model dan memuat bobot ---
    eval_model = build_model(num_classes)
    
    model_filename = f'mbert_lstm_{method}_3class.h5'
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, model_filename)
    try:
        eval_model.load_weights(checkpoint_path)
        print(f"\n✅ Bobot model berhasil dimuat dari: {checkpoint_path}")
    except (OSError, tf.errors.NotFoundError):
        print(f"❌ ERROR: File bobot model tidak ditemukan di '{checkpoint_path}'.")
        return

    # --- Tahap C: Prediksi pada data validasi ---
    X_val_tokens = tokenizer(
        df_val['text'].tolist(), max_length=MAX_LENGTH,
        truncation=True, padding='max_length', return_tensors='tf'
    )
    y_true_encoded = label_encoder.transform(df_val['label'])
    
    print("\n⚙️  Melakukan prediksi pada data validasi...")
    y_pred_probs = eval_model.predict(
        {'input_ids': X_val_tokens.input_ids, 'attention_mask': X_val_tokens.attention_mask}
    )
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)

    # --- Tahap D: Menampilkan Hasil Evaluasi ---
    print("\n" + "="*60)
    print(f"           LAPORAN HASIL KLASIFIKASI ({method.upper()})")
    print("="*60)
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_))
    print("="*60)

    print(f"\n🖼️  Menampilkan Confusion Matrix ({method.upper()})...")
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
    )
    plt.title(f'Confusion Matrix pada Data Validasi ({method.upper()})')
    plt.ylabel('Label Sebenarnya (True)')
    plt.xlabel('Label Prediksi (Predicted)')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan proses evaluasi model.")
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['ros', 'rus', 'ros_enn'],
        help="Metode dari model yang akan dievaluasi: 'ros', 'rus', atau 'ros_enn'."
    )
    args = parser.parse_args()
    main(args.method)