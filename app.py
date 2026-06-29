import os
import json
import threading
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

# ── Load class labels ──
_labels_path = os.path.join(MODELS_DIR, 'class_labels.json')
if os.path.exists(_labels_path):
    with open(_labels_path) as f:
        _labels = json.load(f)
    ALZ_CLASSES   = _labels.get('alzheimer_classes', ['Mild', 'Moderate', 'NonDemented', 'VeryMild'])
    TUMOR_CLASSES = _labels.get('tumor_classes',     ['glioma', 'meningioma', 'notumor', 'pituitary'])
    print(f"Loaded class labels from {_labels_path}", flush=True)
else:
    ALZ_CLASSES   = ['Mild', 'Moderate', 'NonDemented', 'VeryMild']
    TUMOR_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    print("WARNING: class_labels.json not found — using alphabetical fallback.", flush=True)

IMG_SIZE = (224, 224)

# ── Model state ──
alz_model   = None
tumor_model = None
_models_ready  = False          # True once both loaded successfully
_models_error  = None           # Set if loading fails
_loading_lock  = threading.Lock()


def _load_models_background():
    """Runs in a daemon thread so the port binds immediately on startup."""
    global alz_model, tumor_model, _models_ready, _models_error
    print("=== Background model loading started ===", flush=True)
    try:
        with _loading_lock:
            alz_model = load_model(
                os.path.join(MODELS_DIR, 'alzheimer_model.keras'), compile=False)
            print("=== Alzheimer model loaded OK ===", flush=True)

            tumor_model = load_model(
                os.path.join(MODELS_DIR, 'tumor_model.keras'), compile=False)
            print("=== Tumor model loaded OK ===", flush=True)

            _models_ready = True
            print("=== Both models ready — accepting predictions ===", flush=True)
    except Exception as e:
        _models_error = str(e)
        print(f"=== FATAL: model loading failed: {e} ===", flush=True)


# Start loading immediately (non-blocking)
_loader_thread = threading.Thread(target=_load_models_background, daemon=True)
_loader_thread.start()


# ── Helpers ──
def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


LABEL_MAP = {
    'notumor':              'No Tumor',
    'glioma':               'Glioma Tumor',
    'meningioma':           'Meningioma Tumor',
    'pituitary':            'Pituitary Tumor',
    'NonDemented':          'Non-Demented',
    'VeryMild':             'Very Mild Dementia',
    'Mild':                 'Mild Dementia',
    'Moderate':             'Moderate Dementia',
    'nondemented':          'Non-Demented',
    'verymilddemented':     'Very Mild Dementia',
    'milddemented':         'Mild Dementia',
    'moderatedemented':     'Moderate Dementia',
}

def friendly(raw):
    return LABEL_MAP.get(raw, raw)


# ── Routes ──
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":        "healthy",
        "models_ready":  _models_ready,
        "models_error":  _models_error,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # Tell the frontend if models are still warming up
    if not _models_ready:
        if _models_error:
            return jsonify({"error": f"Model loading failed: {_models_error}"}), 500
        return jsonify({"error": "Models are still loading, please retry in a few seconds."}), 503

    file = request.files['image']

    try:
        img           = Image.open(file.stream)
        processed_img = prepare_image(img, IMG_SIZE)

        with _loading_lock:
            alz_pred   = alz_model.predict(processed_img,   verbose=0)[0]
            tumor_pred = tumor_model.predict(processed_img, verbose=0)[0]

        # Sanity-check output sizes
        if len(alz_pred) != len(ALZ_CLASSES):
            return jsonify({"error": (
                f"Alzheimer model has {len(alz_pred)} outputs but "
                f"ALZ_CLASSES has {len(ALZ_CLASSES)} entries. "
                "Fix class_labels.json or ALZ_CLASSES in app.py."
            )}), 500

        if len(tumor_pred) != len(TUMOR_CLASSES):
            return jsonify({"error": (
                f"Tumor model has {len(tumor_pred)} outputs but "
                f"TUMOR_CLASSES has {len(TUMOR_CLASSES)} entries. "
                "Fix class_labels.json or TUMOR_CLASSES in app.py."
            )}), 500

        alz_idx   = int(np.argmax(alz_pred))
        tumor_idx = int(np.argmax(tumor_pred))

        return jsonify({
            "alzheimers_prediction": {
                "label":      friendly(ALZ_CLASSES[alz_idx]),
                "raw_label":  ALZ_CLASSES[alz_idx],
                "confidence": float(alz_pred[alz_idx]),
                "all_scores": {
                    friendly(ALZ_CLASSES[i]): float(alz_pred[i])
                    for i in range(len(ALZ_CLASSES))
                },
            },
            "tumor_prediction": {
                "label":      friendly(TUMOR_CLASSES[tumor_idx]),
                "raw_label":  TUMOR_CLASSES[tumor_idx],
                "confidence": float(tumor_pred[tumor_idx]),
                "all_scores": {
                    friendly(TUMOR_CLASSES[i]): float(tumor_pred[i])
                    for i in range(len(TUMOR_CLASSES))
                },
            },
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
