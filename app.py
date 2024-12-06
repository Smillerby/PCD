import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model('C:/Users/Irpan/Documents/PCD-IRPAN/DATASET/hasil/model_6class.keras')

# Fungsi untuk memproses gambar input
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Mengubah ukuran gambar sesuai input model
    img_array = np.array(img) / 224.0  # Normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
    return img_array

# Fungsi untuk memprediksi gambar
def predict_image(img_array):
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)  # Menentukan kelas dengan probabilitas tertinggi
    return class_idx[0], preds[0][class_idx[0]]  # Mengembalikan label kelas dan probabilitasnya

# Streamlit UI
st.title('Aplikasi Klasifikasi Gambar dengan MobileNet')
st.write('Unggah gambar untuk diklasifikasikan.')

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Proses gambar dan prediksi
    img_array = preprocess_image(uploaded_file)
    class_idx, confidence = predict_image(img_array)
    
    # Menampilkan hasil prediksi
    classes = ['Class 1', 'Class 2', 'Class 3']  # Sesuaikan dengan kelas Anda
    st.write(f'Prediksi: {classes[class_idx]} dengan tingkat kepercayaan {confidence*100:.2f}%')
