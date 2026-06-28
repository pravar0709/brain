import os
import sys
import numpy as np
from flask import Flask, request, jsonify, make_response
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Force CPU before TF touches anything ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# ── CORS: injected manually on every response ──
def corsify(data, status=200):
    r = make_response(jsonify(data) if isinstance(data, dict) else data, status)
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return r

# ── Patch Keras Dense layer to strip unsupported quantization_config ──
_orig_dense_from_config = tf.keras.layers.Dense.from_config

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _orig_dense_from_config.__func__(cls, config)

tf.keras.layers.Dense.from_config = _patched_dense_from_config

# ── Startup diagnostics ──
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
print(f"=== Python        : {sys.version}", flush=True)
print(f"=== TF version    : {tf.__version__}", flush=True)
print(f"=== Models dir    : {MODELS_DIR}", flush=True)
print(f"=== Dir exists    : {os.path.isdir(MODELS_DIR)}", flush=True)

if os.path.isdir(MODELS_DIR):
    files = os.listdir(MODELS_DIR)
    print(f"=== Models found  : {files}", flush=True)
    for f in files:
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"    {f} — {size/1024/1024:.1f} MB", flush=True)
else:
    print("=== FATAL: models/ directory does not exist!", flush=True)
    print("=== Current directory contents:", flush=True)
    for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        print(f"    {f}", flush=True)
    sys.exit(1)

# ── Load models ──
ALZ_MODEL_PATH   = os.path.join(MODELS_DIR, 'alzheimer_model.keras')
TUMOR_MODEL_PATH = os.path.join(MODELS_DIR, 'tumor_model.keras')

if not os.path.exists(ALZ_MODEL_PATH):
    print(f"=== FATAL: alzheimer_model.keras not found at {ALZ_MODEL_PATH}", flush=True)
    sys.exit(1)

if not os.path.exists(TUMOR_MODEL_PATH):
    print(f"=== FATAL: tumor_model.keras not found at {TUMOR_MODEL_PATH}", flush=True)
    sys.exit(1)

print("=== Loading alzheimer_model.keras ...", flush=True)
try:
    alz_model = load_model(ALZ_MODEL_PATH, compile=False)
    print(f"=== Alzheimer model loaded — output shape: {alz_model.output_shape}", flush=True)
except Exception as e:
    print(f"=== FATAL: alzheimer_model failed: {e}", flush=True)
    sys.exit(1)

print("=== Loading tumor_model.keras ...", flush=True)
try:
    tumor_model = load_model(TUMOR_MODEL_PATH, compile=False)
    print(f"=== Tumor model loaded — output shape: {tumor_model.output_shape}", flush=True)
except Exception as e:
    print(f"=== FATAL: tumor_model failed: {e}", flush=True)
    sys.exit(1)

print("=== All models loaded. Flask starting.", flush=True)

# ── Class labels — adjust if your model has more classes ──
ALZ_CLASSES   = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
TUMOR_CLASSES = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']
IMG_SIZE      = (224, 224)

# Auto-detect number of output classes from model
alz_n   = alz_model.output_shape[-1]
tumor_n = tumor_model.output_shape[-1]
print(f"=== Alz model outputs   : {alz_n} classes", flush=True)
print(f"=== Tumor model outputs : {tumor_n} classes", flush=True)

# Trim/pad class lists to match model output
if alz_n == 2:
    ALZ_CLASSES = ['Alzheimer', 'NonAlzheimer']
elif alz_n == 4:
    ALZ_CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
else:
    ALZ_CLASSES = [f'Class_{i}' for i in range(alz_n)]

if tumor_n == 2:
    TUMOR_CLASSES = ['Tumor', 'NoTumor']
elif tumor_n == 4:
    TUMOR_CLASSES = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']
else:
    TUMOR_CLASSES = [f'Class_{i}' for i in range(tumor_n)]

print(f"=== ALZ_CLASSES   : {ALZ_CLASSES}", flush=True)
print(f"=== TUMOR_CLASSES : {TUMOR_CLASSES}", flush=True)


def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# ── Routes ──

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    return corsify({"status": "healthy", "models": "loaded"})


@app.route('/predict', methods=['OPTIONS'])
def predict_preflight():
    return corsify({"ok": True})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        app.logger.error(f"No 'image' key. Keys received: {list(request.files.keys())}")
        return corsify({"error": "No image file uploaded. Use field name 'image'."}, 400)

    file = request.files['image']
    try:
        img           = Image.open(file.stream)
        processed_img = prepare_image(img, IMG_SIZE)

        alz_pred   = alz_model.predict(processed_img, verbose=0)
        tumor_pred = tumor_model.predict(processed_img, verbose=0)

        alz_idx   = int(np.argmax(alz_pred[0]))
        tumor_idx = int(np.argmax(tumor_pred[0]))

        # Build full scores dict for both models
        alz_scores   = {ALZ_CLASSES[i]:   float(alz_pred[0][i])   for i in range(len(ALZ_CLASSES))}
        tumor_scores = {TUMOR_CLASSES[i]: float(tumor_pred[0][i]) for i in range(len(TUMOR_CLASSES))}

        return corsify({
            "alzheimers_prediction": {
                "label":      ALZ_CLASSES[alz_idx],
                "confidence": float(alz_pred[0][alz_idx]),
                "scores":     alz_scores
            },
            "tumor_prediction": {
                "label":      TUMOR_CLASSES[tumor_idx],
                "confidence": float(tumor_pred[0][tumor_idx]),
                "scores":     tumor_scores
            }
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return corsify({"error": str(e)}, 500)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
