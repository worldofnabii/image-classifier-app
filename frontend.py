import streamlit as st
import requests

st.title("AI Image Detector")
st.write("Upload an image to check if it's **Real** or **AI-generated**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded_file})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['label']} ({result['confidence']*100:.2f}%)")
        else:
            st.error("Something went wrong with the backend.")
