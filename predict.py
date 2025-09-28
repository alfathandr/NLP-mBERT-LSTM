# -*- coding: utf-8 -*-
"""
==========================================================================
PREDICT.PY
==========================================================================
Author: Muh. Al Fathan
Project: Analisis Sentimen Code-Mixed mBERT-LSTM

Deskripsi:
-----------
Skrip ini berfungsi sebagai antarmuka interaktif untuk melakukan prediksi
sentimen pada teks baru menggunakan salah satu model yang telah dilatih.

Pengguna dapat memilih model mana yang akan digunakan (berdasarkan metode
balancing data) melalui argumen baris perintah.

Cara Menjalankan (dari Terminal):
----------------------------------
# Untuk melakukan prediksi dengan model hasil training ROS
$ python predict.py --method ros

# Untuk melakukan prediksi dengan model hasil training RUS
$ python predict.py --method rus

# Untuk melakukan prediksi dengan model hasil training ROS+ENN
$ python predict.py --method ros_enn

Setelah berjalan, masukkan kalimat dan tekan Enter. Ketik 'exit' untuk keluar.
==========================================================================
"""

import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model

# --- 1. Konfigurasi ---
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128
SAVED_MODELS_DIR = 'saved_models'


def load_artifacts(method):
    """Memuat tokenizer, label encoder, dan path bobot model yang sesuai."""
    print(f"📖 Memuat artefak untuk metode '{method.upper()}'...")
    try:
        tokenizer_dirname = f'tokenizer_{method}_3class'
        encoder_filename = f'label_encoder_{method}_3class.pkl'
        model_filename = f'mbert_lstm_{method}_3class.h5'
        
        tokenizer_path = os.path.join(SAVED_MODELS_DIR, tokenizer_dirname)
        label_encoder_path = os.path.join(SAVED_MODELS_DIR, encoder_filename)
        checkpoint_path = os.path.join(SAVED_MODELS_DIR, model_filename)
        
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Periksa apakah semua file ada sebelum melanjutkan
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"File bobot model tidak ditemukan di {checkpoint_path}")
            
        print("✅ Artefak berhasil dimuat.")
        return tokenizer, label_encoder, checkpoint_path
        
    except Exception as e:
        print(f"❌ ERROR: Gagal memuat artefak. Pastikan skrip training untuk metode '{method}' sudah dijalankan.")
        print(f"   Detail: {e}")
        return None, None, None


def build_model(num_classes):
    """Membangun ulang arsitektur model yang identik dengan saat training."""
    class BertLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(BertLayer, self).__init__(**kwargs)
            self.bert = TFBertModel.from_pretrained(MODEL_NAME)
            self.bert.trainable = False
        
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
    """Fungsi utama untuk menjalankan pipeline prediksi interaktif."""
    
    # Mematikan log TensorFlow yang kurang penting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # --- Tahap A: Memuat semua yang diperlukan ---
    tokenizer, label_encoder, checkpoint_path = load_artifacts(method)
    if tokenizer is None:
        return
        
    num_classes = len(label_encoder.classes_)
    
    # --- Tahap B: Membangun model dan memuat bobot ---
    prediction_model = build_model(num_classes)
    prediction_model.load_weights(checkpoint_path)
    
    print("\n" + "="*60)
    print(f"🤖 Model '{method.upper()}' siap untuk prediksi!")
    print("   Ketik kalimat Anda lalu tekan Enter.")
    print("   Ketik 'exit' atau 'quit' untuk keluar.")
    print("="*60)
    
    # --- Tahap C: Loop Interaktif ---
    while True:
        try:
            user_input = input("\n👉 Masukkan kalimat: ")
            
            # Kondisi untuk keluar dari loop
            if user_input.lower() in ['exit', 'quit', 'keluar']:
                print("👋 Terima kasih! Sampai jumpa.")
                break

            if not user_input.strip(): # Skip jika input kosong
                continue

            # Tokenisasi teks input dari pengguna
            inputs = tokenizer(
                user_input,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='tf'
            )
            
            # Lakukan prediksi
            probabilities = prediction_model.predict(
                {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask},
                verbose=0 # Matikan log prediksi
            )[0]
            
            # Dapatkan hasil
            predicted_class_index = np.argmax(probabilities)
            predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
            confidence_score = np.max(probabilities)
            
            # Tampilkan hasil ke pengguna
            print("-" * 30)
            print(f"   Prediksi Sentimen: {predicted_class_label}")
            print(f"   Skor Kepercayaan: {confidence_score:.2%}")
            print("-" * 30)
            
        except KeyboardInterrupt: # Handle Ctrl+C
            print("\n👋 Proses dihentikan oleh pengguna. Sampai jumpa.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan antarmuka prediksi interaktif untuk model sentimen.")
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['ros', 'rus', 'ros_enn'],
        help="Metode dari model yang akan digunakan untuk prediksi."
    )
    args = parser.parse_args()
    main(args.method)