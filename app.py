import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
MODEL_PATH = 'kidney_disease_classification_model.h5'  # Make sure this is correct

# Load model (cached)
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# Streamlit UI
st.title("🩺 Kidney Disease Classifier")
st.write("Upload kidney CT images to detect **Cyst**, **Normal**, **Stone**, or **Tumor** conditions.")

# Ask how many images the user wants to upload
num_images = st.number_input("📷 How many images do you want to upload?", min_value=1, max_value=20, value=1, step=1)

# Upload images
uploaded_files = st.file_uploader("📤 Upload your image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != num_images:
        st.warning(f"⚠️ You selected {num_images} image(s), but uploaded {len(uploaded_files)}. Please upload the correct number.")
    else:
        for uploaded_file in uploaded_files:
            st.subheader(f"🖼️ Image: {uploaded_file.name}")

            # Display the image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

            # Preprocess
            img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = np.max(prediction)

            # Display prediction
            st.success(f"🧪 Prediction: **{predicted_class}**")
            st.info(f"📊 Confidence: **{confidence:.2f}**")

            # Show full probabilities (optional)
            with st.expander("🔬 View all class probabilities"):
                for i, prob in enumerate(prediction[0]):
                    st.write(f"{CLASS_NAMES[i]}: {prob:.2f}")
