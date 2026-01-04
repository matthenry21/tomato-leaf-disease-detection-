import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to predict the disease class.")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/tomato_model.h5")
    return model

model = load_model()

# --------------------------------------------------
# Class names (ORDER MUST MATCH train_data.class_indices)
# --------------------------------------------------
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# --------------------------------------------------
# Disease descriptions
# -------------------------------------------------
# Disease descriptions for tomato leaf diseases
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "description": "A bacterial disease causing small, dark, water-soaked spots on leaves.",
        "symptoms": "Leaf spots with yellow halos, defoliation, reduced yield.",
        "management": "Use disease-free seeds, avoid overhead watering, apply copper-based sprays."
    },

    "Tomato___Early_blight": {
        "description": "A common fungal disease caused by Alternaria solani.",
        "symptoms": "Brown spots with concentric rings on older leaves, leaf yellowing.",
        "management": "Remove infected leaves, crop rotation, apply fungicides if needed."
    },

    "Tomato___Late_blight": {
        "description": "A serious disease caused by Phytophthora infestans.",
        "symptoms": "Large dark patches on leaves, white fungal growth under leaves.",
        "management": "Destroy infected plants, improve air circulation, fungicide use."
    },

    "Tomato___Leaf_Mold": {
        "description": "A fungal disease common in humid conditions.",
        "symptoms": "Yellow spots on upper leaf surface, mold growth underneath.",
        "management": "Reduce humidity, ensure ventilation, apply fungicides."
    },

    "Tomato___Septoria_leaf_spot": {
        "description": "A fungal disease that affects older leaves first.",
        "symptoms": "Small circular spots with dark borders and gray centers.",
        "management": "Remove infected leaves, avoid wet foliage, crop rotation."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Damage caused by tiny spider mites feeding on leaves.",
        "symptoms": "Yellow speckling, webbing under leaves, leaf drying.",
        "management": "Spray water regularly, use insecticidal soap or neem oil."
    },

    "Tomato___Target_Spot": {
        "description": "A fungal disease causing target-like lesions.",
        "symptoms": "Brown lesions with concentric rings on leaves.",
        "management": "Remove infected debris, proper spacing, fungicide application."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A viral disease transmitted by whiteflies.",
        "symptoms": "Upward leaf curling, yellowing, stunted growth.",
        "management": "Control whiteflies, remove infected plants."
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": "A viral disease affecting tomato plants.",
        "symptoms": "Mottled light and dark green leaves, distorted growth.",
        "management": "Disinfect tools, remove infected plants."
    },

    "Tomato___healthy": {
        "description": "Healthy tomato leaf with no visible disease.",
        "symptoms": "Green leaves with normal shape and texture.",
        "management": "Maintain good watering and nutrient practices."
    }
}
# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing image...")

    # --------------------------------------------------
    # IMPORTANT: Must match training target_size=(128,128)
    # --------------------------------------------------
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    predicted_label = class_names[predicted_class]
    disease_details = DISEASE_INFO.get(predicted_label, None)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    st.success(f"**Prediction:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2%}")
    
    if disease_details:
        st.subheader("🧬 Disease Information")

        st.markdown(f"**Description:** {disease_details['description']}")
        st.markdown(f"**Symptoms:** {disease_details['symptoms']}")
        st.markdown(f"**Management Tips:** {disease_details['management']}")
