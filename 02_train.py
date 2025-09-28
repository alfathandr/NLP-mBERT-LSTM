# -*- coding: utf-8 -*-
"""
==========================================================================
02_TRAIN.PY
==========================================================================
Author: Muh. Al Fathan
Project: Analisis Sentimen Code-Mixed mBERT-LSTM

Deskripsi:
-----------
Skrip ini bertanggung jawab untuk melatih model klasifikasi sentimen
berdasarkan dataset yang telah diproses oleh `01_data_preparation.py`.
Skrip ini dapat melatih model untuk salah satu dari tiga metode balancing
yang berbeda dengan menentukan argumen `--method`.

Arsitektur yang digunakan adalah model hibrida mBERT-LSTM.

Input:
-------
- Salah satu file CSV seimbang dari `01_data_preparation.py`:
  - `dataset/train_ros_balanced_3class.csv`
  - `dataset/train_rus_balanced_3class.csv`
  - `dataset/train_ros_enn_balanced_3class.csv`

Output:
--------
- Artefak training yang diberi nama sesuai metodenya:
  - Bobot model terbaik (`.h5`).
  - Direktori Tokenizer.
  - File Label Encoder (`.pkl`).

Cara Menjalankan (dari Terminal):
----------------------------------
# Untuk melatih model pada data hasil ROS
$ python 02_train.py --method ros

# Untuk melatih model pada data hasil RUS
$ python 02_train.py --method rus

# Untuk melatih model pada data hasil ROS+ENN
$ python 02_train.py --method ros_enn
==========================================================================
"""

import os
import pickle
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# --- 1. Konfigurasi dan Hyperparameter ---
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Konfigurasi Path
DATASET_DIR = 'dataset'
SAVED_MODELS_DIR = 'saved_models'
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def load_and_split_data(input_csv_path):
    """Memuat dataset yang sudah diproses dan membaginya menjadi set latih & validasi."""
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di '{input_csv_path}'.")
        print("Pastikan Anda sudah menjalankan skrip '01_data_preparation.py' dengan metode yang sesuai.")
        return None, None
    
    df.dropna(inplace=True)
    print(f"📖 Total data yang dimuat untuk pelatihan: {len(df)}")

    df_train, df_val = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df['label']
    )
    return df_train, df_val


def build_model(num_classes):
    """Membangun arsitektur model hibrida mBERT-LSTM."""
    
    class BertLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(BertLayer, self).__init__(**kwargs)
            # Memuat model mBERT pre-trained dari Hugging Face.
            # Dibiarkan tanpa argumen tambahan (seperti from_pt) agar transformers
            # dapat memilih file bobot yang paling optimal secara otomatis.
            self.bert = TFBertModel.from_pretrained(MODEL_NAME)
            self.bert.trainable = True # Melakukan fine-tuning pada mBERT.
        
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
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main(method):
    """Fungsi utama untuk menjalankan pipeline training."""

    print(f"\n===== MEMULAI PROSES TRAINING UNTUK METODE: {method.upper()} =====")
    
    # --- 2. Menentukan Nama File secara Dinamis ---
    input_filename = f'train_{method}_balanced_3class.csv'
    model_filename = f'mbert_lstm_{method}_3class.h5'
    tokenizer_dirname = f'tokenizer_{method}_3class'
    encoder_filename = f'label_encoder_{method}_3class.pkl'
    
    input_path = os.path.join(DATASET_DIR, input_filename)

    # --- 3. Memuat dan Membagi Data ---
    df_train, df_val = load_and_split_data(input_path)
    if df_train is None:
        return

    # --- 4. Label Encoding ---
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(df_train['label'])
    y_val_encoded = label_encoder.transform(df_val['label'])
    num_classes = len(label_encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes=num_classes)
    y_val = to_categorical(y_val_encoded, num_classes=num_classes)
    print(f"✅ Kelas yang ditemukan: {list(label_encoder.classes_)}")

    encoder_path = os.path.join(SAVED_MODELS_DIR, encoder_filename)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder tersimpan di: {encoder_path}")

    # --- 5. Tokenisasi ---
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    X_train = tokenizer(df_train['text'].tolist(), max_length=MAX_LENGTH, truncation=True, padding='max_length', return_tensors='tf')
    X_val = tokenizer(df_val['text'].tolist(), max_length=MAX_LENGTH, truncation=True, padding='max_length', return_tensors='tf')

    tokenizer_path = os.path.join(SAVED_MODELS_DIR, tokenizer_dirname)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"✅ Tokenizer tersimpan di: {tokenizer_path}")
    
    # --- 6. Membangun Model ---
    model = build_model(num_classes)
    model.summary()

    # --- 7. Menyiapkan Callbacks ---
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, model_filename)
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)

    # --- 8. Memulai Pelatihan ---
    print(f"\n--- 🚀 MEMULAI PELATIHAN MODEL (DATA {method.upper()}) ---")
    model.fit(
        {'input_ids': X_train.input_ids, 'attention_mask': X_train.attention_mask},
        y_train,
        validation_data=({'input_ids': X_val.input_ids, 'attention_mask': X_val.attention_mask}, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[model_checkpoint, early_stopping]
    )
    print(f"\n🎉 Pelatihan model selesai! Model terbaik disimpan di: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan proses training model.")
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['ros', 'rus', 'ros_enn'],
        help="Metode balancing dari data yang akan dilatih: 'ros', 'rus', atau 'ros_enn'."
    )
    args = parser.parse_args()
    main(args.method)