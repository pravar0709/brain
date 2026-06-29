import os
import json
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
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Patch Dense layer (handles quantization_config from older saved models) ──
original_dense_from_config = tf.keras.layers.Dense.from_config

@classmethod
def patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return original_dense_from_config.__func__(cls, config)

tf.keras.layers.Dense.from_config = patched_dense_from_config

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ── Load class labels from the JSON the notebook saves ──
# Falls back to the correct Keras ImageDataGenerator folder-name order if file not found.
_labels_path = os.path.join(MODELS_DIR, 'class_labels.json')
if os.path.exists(_labels_path):
    with open(_labels_path) as f:
        _labels = json.load(f)
    ALZ_CLASSES   = _labels.get('alzheimer_classes',
                                ['Mild', 'Moderate', 'NonDemented', 'VeryMild'])
    TUMOR_CLASSES = _labels.get('tumor_classes',
                                ['glioma', 'meningioma', 'notumor', 'pituitary'])
    print(f"Loaded class labels from {_labels_path}")
else:
    # Keras ImageDataGenerator sorts folder names alphabetically — match that here.
    ALZ_CLASSES   = ['Mild', 'Moderate', 'NonDemented', 'VeryMild']
    TUMOR_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    print("WARNING: class_labels.json not found — using alphabetical fallback class names.")
    print("  ALZ_CLASSES   =", ALZ_CLASSES)
    print("  TUMOR_CLASSES =", TUMOR_CLASSES)
    print("  If these don't match your training folders, predictions will be wrong.")

IMG_SIZE = (224, 224)

# ── Model cache ──
alz_model   = None
tumor_model = None

def get_models():
    """Lazy-loads models on first request so the app starts fast on Render."""
    global alz_model, tumor_model
    if alz_model is None or tumor_model is None:
        print("=== Lazy-loading models (first request only) ===", flush=True)
        try:
            alz_model = load_model(
                os.path.join(MODELS_DIR, 'alzheimer_model.keras'), compile=False)
            print("=== Alzheimer model loaded OK ===", flush=True)
        except Exception as e:
            print(f"=== FATAL: alzheimer_model failed to load: {e}", flush=True)
        try:
            tumor_model = load_model(
                os.path.join(MODELS_DIR, 'tumor_model.keras'), compile=False)
            print("=== Tumor model loaded OK ===", flush=True)
        except Exception as e:
            print(f"=== FATAL: tumor_model failed to load: {e}", flush=True)
    return alz_model, tumor_model


def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)   # EfficientNet-specific normalisation


def make_friendly_label(raw_label: str) -> str:
    """
    Converts folder-style names to human-readable ones.
    e.g. 'notumor' -> 'No Tumor', 'NonDemented' -> 'Non-Demented'
    """
    mapping = {
        # Tumor classes
        'notumor':    'No Tumor',
        'glioma':     'Glioma Tumor',
        'meningioma': 'Meningioma Tumor',
        'pituitary':  'Pituitary Tumor',
        # Alzheimer classes
        'nondemented':   'Non-Demented',
        'veryMildDemented': 'Very Mild Dementia',
        'mildDemented':     'Mild Dementia',
        'moderateDemented': 'Moderate Dementia',
        # Common alternate spellings
        'VeryMild': 'Very Mild Dementia',
        'Mild':     'Mild Dementia',
        'Moderate': 'Moderate Dementia',
        'NonDemented': 'Non-Demented',
    }
    return mapping.get(raw_label, raw_label)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']

    try:
        a_model, t_model = get_models()
        if a_model is None or t_model is None:
            return jsonify({"error": "One or both models failed to load on the server."}), 500

        img           = Image.open(file.stream)
        processed_img = prepare_image(img, IMG_SIZE)

        # ── Run both models ──
        alz_pred   = a_model.predict(processed_img, verbose=0)[0]   # shape: (num_alz_classes,)
        tumor_pred = t_model.predict(processed_img, verbose=0)[0]   # shape: (num_tumor_classes,)

        # Guard: number of outputs must match our class list lengths
        if len(alz_pred) != len(ALZ_CLASSES):
            return jsonify({
                "error": (
                    f"Alzheimer model outputs {len(alz_pred)} classes but "
                    f"ALZ_CLASSES has {len(ALZ_CLASSES)} entries. "
                    "Update ALZ_CLASSES in app.py or regenerate class_labels.json."
                )
            }), 500

        if len(tumor_pred) != len(TUMOR_CLASSES):
            return jsonify({
                "error": (
                    f"Tumor model outputs {len(tumor_pred)} classes but "
                    f"TUMOR_CLASSES has {len(TUMOR_CLASSES)} entries. "
                    "Update TUMOR_CLASSES in app.py or regenerate class_labels.json."
                )
            }), 500

        alz_idx   = int(np.argmax(alz_pred))
        tumor_idx = int(np.argmax(tumor_pred))

        alz_raw_label   = ALZ_CLASSES[alz_idx]
        tumor_raw_label = TUMOR_CLASSES[tumor_idx]

        return jsonify({
            "alzheimers_prediction": {
                "label":      make_friendly_label(alz_raw_label),
                "raw_label":  alz_raw_label,
                "confidence": float(alz_pred[alz_idx]),
                # Full score breakdown so the frontend (or debugging) can show all classes
                "all_scores": {
                    make_friendly_label(ALZ_CLASSES[i]): float(alz_pred[i])
                    for i in range(len(ALZ_CLASSES))
                }
            },
            "tumor_prediction": {
                "label":      make_friendly_label(tumor_raw_label),
                "raw_label":  tumor_raw_label,
                "confidence": float(tumor_pred[tumor_idx]),
                "all_scores": {
                    make_friendly_label(TUMOR_CLASSES[i]): float(tumor_pred[i])
                    for i in range(len(TUMOR_CLASSES))
                }
            }
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
