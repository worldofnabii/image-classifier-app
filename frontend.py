import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import sqlite3
import os
import datetime

# =============================
# Database Setup
# =============================
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_prediction(filename, label, confidence):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (filename, label, confidence, timestamp) VALUES (?, ?, ?, ?)", 
              (filename, label, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_predictions():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("umm-maybe/AI-image-detector")
    model = AutoModelForImageClassification.from_pretrained(
        "umm-maybe/AI-image-detector", torch_dtype="auto"
    )
    return extractor, model

extractor, model = load_model()

# =============================
# Streamlit UI
# =============================
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

        # Save prediction to database
        insert_prediction(uploaded_file.name, label, confidence)

# =============================
# Show Prediction History
# =============================
st.subheader("Prediction History")
rows = fetch_predictions()

if rows:
    for row in rows:
        st.write(f"🖼️ {row[1]} → {row[2]} ({row[3]*100:.2f}%) at {row[4]}")
else:
    st.info("No predictions stored yet.")

