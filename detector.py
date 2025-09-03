import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

# Load model + feature extractor
extractor = AutoFeatureExtractor.from_pretrained("umm-maybe/AI-image-detector")
model = AutoModelForImageClassification.from_pretrained("umm-maybe/AI-image-detector", torch_dtype="auto")

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    label = model.config.id2label[pred_id]
    return {"label": label, "confidence": round(confidence, 3)}
