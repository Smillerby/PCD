import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set path to model
model_path = 'C:/Users/Irpan/Documents/PCD-IRPAN/DATASET/hasil/model_6class.keras'

# Load the model
model = tf.keras.models.load_model(model_path)

# Set image size
IMG_SIZE = 224

# Class names (adjust these to your own class names)
class_names = ['Class1', 'Class2', 'Class3']  # Ganti dengan nama kelas yang sesuai

# Define image processing function
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale image
    return image

# Define prediction function
def predict_image(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions

# Streamlit UI setup
st.title("Image Classification with MobileNet")
st.write("Upload an image to classify it using a pre-trained MobileNet model.")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the class
    predicted_class, predictions = predict_image(image)
    
    st.write(f"Prediction: {class_names[predicted_class[0]]}")
    st.write(f"Prediction probabilities: {predictions}")

    # Show a bar chart of probabilities
    st.write("Prediction Probabilities:")
    prob_fig = plt.figure()
    sns.barplot(x=class_names, y=predictions[0])
    plt.title("Prediction Probabilities")
    st.pyplot(prob_fig)

# For confusion matrix and classification report, you would need the test data
# to evaluate the model. This is optional and depends on your use case.

# If you want to evaluate model performance using a sample set:
if st.button('Evaluate Model'):
    test_dir = 'C:/Users/Irpan/Documents/PCD-IRPAN/DATASET/test'
    test_gen = ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_data.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    st.write("Confusion Matrix:")
    cm_fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(),
                yticklabels=test_data.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(cm_fig)

    # Classification Report
    st.write("Classification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=test_data.class_indices.keys())
    st.text(report)

