import os

# =====================================
# MODE APLIKASI
# =====================================
# MODE = "local"   → untuk testing localhost
# MODE = "online"  → untuk server (gunicorn)
MODE = os.getenv("FLASK_MODE", "online")

# =====================================
# KONFIGURASI ENV TENSORFLOW
# =====================================
# Sembunyikan warning TensorFlow (INFO, WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paksa TensorFlow CPU-only (hemat RAM, WAJIB di server kecil)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

# ==========================
# KONFIGURASI DASAR
# ==========================

# Path folder saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Keras (fallback / local)
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# Model TensorFlow Lite (hasil convert + quantization)
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "model_fp16.tflite")

# Ukuran input gambar (HARUS sama dengan training)
IMG_SIZE = (128, 128)

# Nama kelas (URUTAN HARUS SAMA SAAT TRAINING)
CLASS_NAMES = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro"
]

# ==========================
# DETEKSI MODE MODEL
# ==========================
# Jika file .tflite ada → pakai TFLite (lebih ringan & hemat RAM)
# Jika tidak ada → fallback ke model.h5 (keras)
USE_TFLITE = os.path.exists(TFLITE_MODEL_PATH)

# ==========================
# LOAD MODEL (SEKALI SAJA)
# ==========================

if USE_TFLITE:
    print(f"[INFO] Loading TFLite model from: {TFLITE_MODEL_PATH}")

    # Inisialisasi interpreter TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Ambil detail input & output tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Keras model tidak dipakai
    model = None

    print("[INFO] TFLite model loaded successfully.")

else:
    print(f"[INFO] Loading Keras model from: {MODEL_PATH}")

    # Load model Keras (.h5)
    model = tf.keras.models.load_model(MODEL_PATH)

    # Interpreter tidak dipakai
    interpreter = None

    print("[INFO] Keras model loaded successfully.")

# ==========================
# PREPROCESS IMAGE
# ==========================
def preprocess_image(image_bytes):
    """
    Preprocessing gambar:
    - Decode bytes → OpenCV image
    - Resize ke IMG_SIZE
    - Normalisasi ke range [0,1]
    - Tambah dimensi batch
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Gambar tidak valid (JPG/PNG saja).")

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ==========================
# FUNGSI PREDICT UNIVERSAL
# ==========================
def predict_model(img):
    """
    Fungsi prediksi universal:
    - Jika TFLite → pakai interpreter
    - Jika Keras → pakai model.predict
    """
    if USE_TFLITE:
        interpreter.set_tensor(
            input_details[0]["index"],
            img.astype(np.float32)
        )
        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )
        return output[0]

    else:
        return model.predict(img)[0]

# ==========================
# FLASK APP
# ==========================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint health-check
    Digunakan untuk:
    - Test server hidup
    - Monitoring deployment
    """
    return jsonify({
        "status": "ok",
        "mode": MODE,
        "model": "tflite" if USE_TFLITE else "keras",
        "message": "API is running"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama prediksi penyakit daun padi
    Input  : file gambar (field name: image)
    Output : label, confidence, probabilities
    """
    if "image" not in request.files:
        return jsonify({"error": "Field 'image' tidak ditemukan."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    try:
        # Preprocess gambar
        img = preprocess_image(file.read())

        # Prediksi (AUTO keras / tflite)
        preds = predict_model(img)

        # Ambil kelas dengan probabilitas tertinggi
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        return jsonify({
            "label": CLASS_NAMES[idx],
            "confidence": confidence,
            "probs": {
                CLASS_NAMES[i]: float(p)
                for i, p in enumerate(preds)
            }
        }), 200

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================
# JALANKAN SERVER
# ==========================

if __name__ == "__main__":

    # ===============================
    # MODE LOCAL (UNCOMMENT JIKA TEST)
    # ===============================
    if MODE == "local":
        print("[INFO] Running in LOCAL mode")
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=True
        )

    # ==================================
    # MODE ONLINE (GUNICORN AKAN HANDLE)
    # ==================================
    else:
        print("[INFO] Running in ONLINE mode via Gunicorn")
