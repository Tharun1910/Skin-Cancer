import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf

@st.cache_resource
def load_resnet_model():
    try:
        resnet_model = load_model("models/resnet_model.h5")
        return resnet_model
    except FileNotFoundError:
        st.error("ResNet50 model file not found. Please ensure the file 'resnet_model.h5' exists.")
        st.stop()

# Preprocess the uploaded image for ResNet50
def preprocess_image_resnet(img_path, size=(224, 224)):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return arr

# Streamlit app layout
st.title("Skin Cancer Classification with ResNet50")
st.write("Upload an image, and the app will classify it using ResNet50.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    img_path = f"temp_image.{uploaded_file.type.split('/')[-1]}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the ResNet50 model
    resnet_model = load_resnet_model()

    # Define class labels (replace with your actual labels)
    class_labels = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

    # Preprocess the image
    img_array_resnet = preprocess_image_resnet(img_path)

    # Run ResNet50 prediction
    preds_resnet = resnet_model.predict(img_array_resnet)
    pred_idx_resnet = np.argmax(preds_resnet, axis=1)[0]
    pred_label_resnet = class_labels[pred_idx_resnet]
    confidence_resnet = preds_resnet[0][pred_idx_resnet]

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    # Display ResNet50 prediction
    st.subheader("ResNet50 Prediction")
    st.write(f"Predicted Class: {pred_label_resnet}")
    st.write(f"Confidence: {confidence_resnet:.2f}")

    # Clean up temporary file
    os.remove(img_path)