# server/models/use_model.py

import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path


MODEL_PATH = Path("server/models/images/final_inceptionresnetv3_art_model_v5.h5")
CLASS_PATH = Path("server/models/images/class_indices.json")

model = load_model(MODEL_PATH, safe_mode=False)
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx_to_class = json.load(f)
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

def run_prediction_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    preds = model.predict(img_array)

    predicted_class_idx = np.argmax(preds)
    predicted_artist = idx_to_class.get(predicted_class_idx, "unknown")
    confidence = float(np.max(preds))

    return {"artist": predicted_artist, "confidence": confidence}
