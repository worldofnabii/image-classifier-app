# AI Image Detector

This project is a simple **image classification app** that detects whether an image is **Real** or **AI-generated**.  

It is built with:
- [Streamlit](https://streamlit.io/) for the frontend UI
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the AI model
- [PyTorch](https://pytorch.org/) for deep learning


# Live Demo
Try it out here: [Streamlit Cloud App](https://share.streamlit.io/Worldofnabii/image-classifier-app/main/frontend.py)  


# Project Structure
ImageSegmentation/
│── frontend.py # Main Streamlit app
│── requirements.txt # Dependencies
│── .gitignore # Ignored files (uploads, cache, etc.)
└── README.md # Project description


## ⚙️ Setup & Installation (Run Locally)
1. Clone this repository:
   ```bash
   git clone https://github.com/Worldofnabii/image-classifier-app.git
   cd image-classifier-app
(Optional) Create and activate a virtual environment:

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install dependencies:

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run frontend.py

# Usage
Upload an image (.jpg, .jpeg, .png).

Click Predict.

The app will classify the image as Real or AI-generated, with a confidence score.

# Notes
The app uses the Hugging Face model: umm-maybe/AI-image-detector.

Flask backend was removed for simplicity; everything runs directly in Streamlit.

Temporary files (uploads, cache, etc.) are ignored from GitHub for a cleaner repo.

# Author
Worldofnabii (2025)


