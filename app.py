import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
# Enable CORS for all routes automatically
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Patch Dense layer ──
original_dense_from_config = tf.keras.layers.Dense.from_config

@classmethod
def patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return original_dense_from_config.__func__(cls, config)

tf.keras.layers.Dense.from_config = patched_dense_from_config

# ── Load models with explicit error logging ──
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

print("=== Loading alzheimer model...", flush=True)
try:
    alz_model = load_model(
        os.path.join(MODELS_DIR, 'alzheimer_model.keras'), compile=False
    )
    print("=== Alzheimer model loaded OK", flush=True)
except Exception as e:
    print(f"=== FATAL: alzheimer_model failed to load: {e}", flush=True)
    sys.exit(1)

print("=== Loading tumor model...", flush=True)
try:
    tumor_model = load_model(
        os.path.join(MODELS_DIR, 'tumor_model.keras'), compile=False
    )
    print("=== Tumor model loaded OK", flush=True)
except Exception as e:
    print(f"=== FATAL: tumor_model failed to load: {e}", flush=True)
    sys.exit(1)

ALZ_CLASSES   = ['Alzheimer', 'Tumor']
TUMOR_CLASSES = ['Alzheimer', 'Tumor']
IMG_SIZE      = (224, 224)

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    try:
        img           = Image.open(file.stream)
        processed_img = prepare_image(img, IMG_SIZE)
        alz_pred      = alz_model.predict(processed_img)
        tumor_pred    = tumor_model.predict(processed_img)
        alz_idx       = np.argmax(alz_pred[0])
        tumor_idx     = np.argmax(tumor_pred[0])

        return jsonify({
            "alzheimers_prediction": {
                "label":      ALZ_CLASSES[alz_idx],
                "confidence": float(alz_pred[0][alz_idx])
            },
            "tumor_prediction": {
                "label":      TUMOR_CLASSES[tumor_idx],
                "confidence": float(tumor_pred[0][tumor_idx])
            }
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
