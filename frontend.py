import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from dotenv import load_dotenv
import os
from mysql.connector import pooling, Error
import datetime
import io

# ---- Load .env ----
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", ""),
    "database": os.getenv("DB_NAME", "image_classification"),
}

# ---- Create uploads folder ----
os.makedirs("uploads", exist_ok=True)

# ---- DB pool (cached resource) ----
@st.cache_resource
def init_pool():
    try:
        pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=4, **DB_CONFIG)
        return pool
    except Error as e:
        st.error(f"DB Pool init error: {e}")
        return None

def get_conn():
    pool = init_pool()
    if pool is None:
        raise RuntimeError("DB pool not initialized.")
    return pool.get_connection()

def save_prediction(filename: str, prediction: str, confidence: float):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        sql = "INSERT INTO image_records (filename, prediction, confidence, timestamp) VALUES (%s, %s, %s, NOW())"
        cur.execute(sql, (filename, prediction, float(confidence)))
        conn.commit()
        last_id = cur.lastrowid
        cur.close()
        return last_id
    except Exception as e:
        st.error(f"Error saving to DB: {e}")
        return None
    finally:
        if conn is not None and conn.is_connected():
            conn.close()

def fetch_recent(limit: int = 100):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, filename, prediction, confidence, timestamp FROM image_records ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()  # list of dicts
        cur.close()
        return rows
    except Exception as e:
        st.error(f"Error fetching from DB: {e}")
        return []
    finally:
        if conn is not None and conn.is_connected():
            conn.close()

# ---- Load model (unchanged) ----
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("umm-maybe/AI-image-detector")
    model = AutoModelForImageClassification.from_pretrained(
        "umm-maybe/AI-image-detector", torch_dtype="auto"
    )
    return extractor, model

extractor, model = load_model()

# ---- Streamlit UI ----
st.title("AI Image Detector")
st.write("Upload an image to check if it's **Real** or **AI-generated**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # save uploaded file locally (so filename persists)
    fname = uploaded_file.name
    save_path = os.path.join("uploads", fname)
    # ensure unique filename if needed
    if os.path.exists(save_path):
        # add timestamp to avoid overwrite
        base, ext = os.path.splitext(fname)
        fname = f"{base}_{int(datetime.datetime.now().timestamp())}{ext}"
        save_path = os.path.join("uploads", fname)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Predict"):
        # run model
        try:
            inputs = extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_id].item()
            label = model.config.id2label[pred_id]
            st.success(f"Prediction: {label} ({confidence*100:.2f}%)")
            # save to MySQL
            inserted_id = save_prediction(fname, label, confidence)
            if inserted_id:
                st.write(f"Saved prediction id: {inserted_id}")
        except Exception as e:
            st.error(f"Prediction error: {e}")


