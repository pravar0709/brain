import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)  # Allows your HTML frontend to talk to this API safely

# --- ADDED: Custom Object Scope to ignore quantization_config ---
@tf.keras.utils.register_keras_serializable()
class FixedDense(tf.keras.layers.Dense):
    def get_config(self):
        config = super().get_config()
        config.pop('quantization_config', None)
        return config

custom_objects = {'Dense': FixedDense}
# ----------------------------------------------------------------

# 1. Load your trained models with the custom scope
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

with tf.keras.utils.custom_object_scope(custom_objects):
    alz_model = load_model(os.path.join(MODELS_DIR, 'alzheimer_model.keras'))
    tumor_model = load_model(os.path.join(MODELS_DIR, 'tumor_model.keras'))

# Define classes matching your exact training variables
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
        
        # Run predictions across both models
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
    # Use environment port for Render or default locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
