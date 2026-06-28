import os
import json
import numpy as np
import h5py
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import model_from_config
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

# --- ROBUST MODEL LOADING FIX ---
def robust_load_model(model_path):
    """
    Manually loads model config, strips unsupported quantization keys,
    and then loads weights to bypass deserialization errors.
    """
    with h5py.File(model_path, 'r') as f:
        # 1. Get the model configuration string
        model_config_str = f.attrs.get('model_config')
        if not model_config_str:
            raise ValueError("Model file does not contain a valid configuration.")
            
        model_config = json.loads(model_config_str)

        # 2. Recursive function to strip 'quantization_config' from all layers
        def strip_quantization(obj):
            if isinstance(obj, dict):
                obj.pop('quantization_config', None)
                for key in obj:
                    strip_quantization(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    strip_quantization(item)
        
        strip_quantization(model_config)
        
        # 3. Reconstruct model structure from the cleaned config
        model = model_from_config(model_config)
        
        # 4. Load the weights into the reconstructed model
        model.load_weights(model_path)
        return model

# Load models safely
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
alz_model = robust_load_model(os.path.join(MODELS_DIR, 'alzheimer_model.keras'))
tumor_model = robust_load_model(os.path.join(MODELS_DIR, 'tumor_model.keras'))

# Define classes matching your training variables
ALZ_CLASSES = ['Alzheimer', 'Tumor']
TUMOR_CLASSES = ['Alzheimer', 'Tumor']
IMG_SIZE = (224, 224)

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    
    file = request.files['image']
    try:
        img = Image.open(file.stream)
        processed_img = prepare_image(img, IMG_SIZE)
        
        alz_pred = alz_model.predict(processed_img)
        tumor_pred = tumor_model.predict(processed_img)
        
        alz_idx = np.argmax(alz_pred[0])
        tumor_idx = np.argmax(tumor_pred[0])
        
        return jsonify({
            "alzheimers_prediction": {
                "label": ALZ_CLASSES[alz_idx],
                "confidence": float(alz_pred[0][alz_idx])
            },
            "tumor_prediction": {
                "label": TUMOR_CLASSES[tumor_idx],
                "confidence": float(tumor_pred[0][tumor_idx])
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
