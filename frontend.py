import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

# Load model + feature extractor once
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("umm-maybe/AI-image-detector")
    model = AutoModelForImageClassification.from_pretrained(
        "umm-maybe/AI-image-detector", torch_dtype="auto"
    )
    return extractor, model

extractor, model = load_model()

# Streamlit UI
st.title("AI Image Detector")
st.write("Upload an image to check if it's **Real** or **AI-generated**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        inputs = extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        label = model.config.id2label[pred_id]
        st.success(f"Prediction: {label} ({confidence*100:.2f}%)")

