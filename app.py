from flask import Flask, request, jsonify
from detector import predict_image
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask server is running. Use POST /predict to send an image."


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    result = predict_image(filepath)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
