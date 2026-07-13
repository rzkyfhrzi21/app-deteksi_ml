# ============================================================
# FILE: api_flask.py
# TUJUAN: Server API utama untuk menerima gambar daun padi
#         dari aplikasi PHP, memprosesnya dengan model AI,
#         dan mengembalikan hasil prediksi dalam format JSON.
#
# Ibarat "dokter otomatis" — foto daun padi dikirim ke sini,
# lalu model CNN memeriksa dan memberikan diagnosis penyakitnya.
# Pengguna tidak langsung berinteraksi dengan file ini.
# Yang berinteraksi adalah aplikasi PHP (melalui cURL).
#
# ENDPOINT YANG TERSEDIA:
#   GET  /health  → Cek apakah server masih hidup (dipakai ping_render.php)
#   POST /predict → Kirimi gambar daun padi, terima hasil prediksi JSON
#
# ALUR KERJA KESELURUHAN:
# (1) Server dinyalakan → file model TFLite dibaca dan dimuat ke RAM
# (2) Aplikasi PHP (function_deteksi.php) mengirim gambar via POST ke /predict
# (3) Gambar dipreproses: resize 128x128, ubah warna BGR→RGB, normalisasi 0-1
# (4) Model CNN "memeriksa" gambar dan mengeluarkan 5 angka probabilitas
# (5) Angka tertinggi = penyakit yang diprediksi, dikembalikan sebagai JSON
#
# STRUKTUR FOLDER:
#   app-deteksi_ml/
#   ├── api_flask.py          ← file ini (server API)
#   ├── model/
#   │   ├── model_fp16.tflite ← model UTAMA yang dipakai (lebih ringan, ~17MB)
#   │   ├── best_model.h5     ← model Keras asli (backup, ~100MB)
#   │   └── model_final.h5   ← model Keras epoch terakhir (backup)
#   ├── convert_to_tflite.py  ← script konversi H5 → TFLite
#   └── train_colab.py        ← script training di Google Colab
# ============================================================
import os

# =====================================
# MODE APLIKASI
# =====================================
# MODE = "local"   → untuk testing localhost
# MODE = "online"  → untuk server (gunicorn)
# Nilai diambil dari environment variable FLASK_MODE.
# Jika tidak diset, default ke "online" (aman untuk production).
MODE = os.getenv("FLASK_MODE", "online")

# =====================================
# KONFIGURASI ENV TENSORFLOW
# =====================================
# Sembunyikan warning TensorFlow (INFO, WARNING) agar log server tidak penuh
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paksa TensorFlow hanya pakai CPU (bukan GPU)
# Wajib di server kecil seperti Render.com free karena tidak ada GPU
# Ini juga menghemat RAM secara signifikan
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

# ==========================
# KONFIGURASI DASAR
# ==========================

# Path folder tempat file api_flask.py ini berada (root dari repo app-deteksi_ml)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# PATH FILE MODEL
# Semua file model disimpan di subfolder model/ agar lebih rapi.
# Jika Anda memindahkan folder model, cukup ubah MODEL_DIR saja di sini.
# ===========================================================================

# Folder tempat semua file model disimpan
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Path ke model TFLite FP16 — INI YANG DIPAKAI UTAMA di server production
# Ukuran ~17 MB, hasil konversi dari best_model.h5, akurasi hampir sama
# Lebih ringan dari H5 karena bilangan desimal dikompres dari 32-bit → 16-bit
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model_fp16.tflite")

# Path ke model Keras H5 — dipakai sebagai CADANGAN jika TFLite tidak ada
# Ukuran ~100 MB, butuh RAM jauh lebih besar, lebih lambat dimuat
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

# Ukuran input gambar — HARUS sama persis dengan saat training model
# Semua gambar akan di-resize ke 128x128 piksel sebelum dimasukkan ke model
IMG_SIZE = (128, 128)

# Nama kelas output model — URUTAN HARUS SAMA PERSIS SEPERTI class_indices saat training
# Urutan ini dikunci via parameter classes= di flow_from_directory() di train_colab.py
# Jangan diubah urutan ini! Index 0 = Healthy, bukan yang lain.
CLASS_NAMES = [
    "Healthy",          # index 0 → Daun sehat, tidak ada penyakit
    "Bacterialblight",  # index 1 → Penyakit hawar bakteri
    "Blast",            # index 2 → Penyakit blas / busuk leher
    "Brownspot",        # index 3 → Penyakit bercak cokelat
    "Tungro",           # index 4 → Penyakit tungro (virus)
]

# ==========================
# DETEKSI MODE MODEL (OTOMATIS)
# ==========================
# Sistem otomatis mengecek: apakah file model_fp16.tflite sudah ada di folder model/?
# Jika ADA  → pakai TFLite (lebih ringan, hemat RAM ~4x, akurasi hampir sama)
# Jika TIDAK → fallback otomatis ke best_model.h5 (Keras, ukuran besar tapi lebih portabel)
#
# Anda tidak perlu mengubah kode ini. Cukup pastikan file .tflite ada di folder model/.
USE_TFLITE = os.path.exists(TFLITE_MODEL_PATH)

# ==========================
# LOAD MODEL (SEKALI SAJA SAAT SERVER DINYALAKAN)
# ==========================
# Model hanya dimuat SEKALI saat server pertama kali hidup, bukan setiap ada request masuk.
# Ini sangat penting untuk efisiensi — bayangkan jika harus membaca file 17MB setiap
# kali pengguna mengirim foto! Server akan sangat lambat dan boros RAM.
#
# Cara kerjanya: saat Gunicorn/Python menjalankan file ini, kode di bawah dieksekusi
# langsung, dan hasilnya (variabel interpreter/model) disimpan di RAM untuk dipakai
# berkali-kali oleh fungsi predict().

if USE_TFLITE:
    print(f"[INFO] Memuat model TFLite dari: {TFLITE_MODEL_PATH}")

    # Buat interpreter TFLite — ibarat 'menyiapkan mesin prediksi' dan mengalokasikan memorinya
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()  # Siapkan slot memori untuk data input dan output model

    # Ambil informasi tentang slot input dan output model:
    # input_details  → tahu format gambar yang diharapkan model (ukuran, tipe data)
    # output_details → tahu format hasil yang dikeluarkan model (berapa kelas, tipe data)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model = None  # Variabel model Keras tidak dipakai saat TFLite aktif
    print("[INFO] Model TFLite berhasil dimuat!")

else:
    print(f"[INFO] TFLite tidak ditemukan. Memuat model Keras dari: {MODEL_PATH}")

    # Muat model Keras (.h5) — ukurannya lebih besar (~100MB) tapi cara pakainya lebih mudah
    model = tf.keras.models.load_model(MODEL_PATH)

    interpreter = None  # Variabel interpreter TFLite tidak dipakai saat mode Keras aktif
    print("[INFO] Model Keras berhasil dimuat!")

# ==========================
# FUNGSI: PREPROCESS IMAGE
# ==========================
def preprocess_image(image_bytes):
    """
    Mempersiapkan gambar agar siap dimasukkan ke model AI.

    Ibarat 'memasak bahan makanan' sebelum dimasak — gambar mentah dari
    pengguna perlu disiapkan dulu agar formatnya cocok dengan yang diharapkan model.

    ALUR KERJA (harus IDENTIK dengan preprocessing saat training):
    (1) Ubah data bytes → gambar OpenCV (format BGR)
    (2) Validasi: pastikan gambar berhasil dibaca
    (3) [KRITIKAL] Konversi warna BGR → RGB
        (OpenCV membaca BGR, tapi model dilatih dengan gambar RGB!
         Tanpa konversi ini, warna gambar terbalik dan prediksi AKAN SALAH)
    (4) Resize gambar ke ukuran 128x128 piksel (sesuai input model)
    (5) Normalisasi nilai piksel: dari range 0-255 → 0.0-1.0
        (Model bekerja lebih baik dengan angka kecil antara 0 dan 1)
    (6) Tambah dimensi batch: dari (128,128,3) → (1,128,128,3)
        (Model mengharapkan input berupa 'kumpulan gambar', bukan 1 gambar saja)
    """
    # (1) Ubah bytes mentah → array numpy → gambar OpenCV (format BGR)
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # OpenCV membaca dalam format BGR

    # (2) Validasi: jika gambar tidak terbaca (format tidak didukung atau file rusak)
    if img is None:
        raise ValueError("Gambar tidak valid (JPG/PNG saja).")

    # (3) [KRITIKAL] Konversi BGR → RGB agar sama dengan training
    # Keras ImageDataGenerator.flow_from_directory() membaca gambar sebagai RGB
    # OpenCV membaca sebagai BGR → tanpa konversi ini, prediksi akan salah!
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMG_SIZE)             # (4) Resize ke 128x128 piksel
    img = img.astype("float32") / 255.0         # (5) Normalisasi: 0-255 → 0.0-1.0
    img = np.expand_dims(img, axis=0)           # (6) Tambah dimensi batch: (128,128,3) → (1,128,128,3)

    return img

# ==========================
# FUNGSI: PREDICT MODEL (UNIVERSAL)
# ==========================
def predict_model(img):
    """
    Fungsi prediksi yang bekerja untuk DUA jenis model (TFLite atau Keras).

    Ibarat 'mesin pendeteksi' yang bisa bekerja dengan dua jenis bahan bakar —
    secara otomatis memilih cara yang tepat berdasarkan model yang tersedia.

    Hasilnya: array 5 angka (probabilitas), contoh:
    [0.02, 0.93, 0.02, 0.02, 0.01]
    → Index 1 (Bacterialblight) paling tinggi = model mendeteksi hawar bakteri
    """
    if USE_TFLITE:
        # === CARA PREDIKSI DENGAN TFLITE INTERPRETER ===
        # (1) Masukkan gambar ke slot input tensor model
        interpreter.set_tensor(
            input_details[0]["index"],  # Index slot input yang benar
            img.astype(np.float32)      # Pastikan tipe data float32 sesuai kebutuhan model
        )
        # (2) Jalankan proses prediksi (model 'berpikir')
        interpreter.invoke()

        # (3) Ambil hasil prediksi dari slot output tensor
        output = interpreter.get_tensor(
            output_details[0]["index"]  # Index slot output yang benar
        )
        return output[0]  # Ambil baris pertama (karena batch size = 1)

    else:
        # === CARA PREDIKSI DENGAN MODEL KERAS ===
        # Lebih sederhana — cukup panggil .predict() dan ambil baris pertama
        return model.predict(img)[0]

# ==========================
# FLASK APP
# ==========================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    """
    ENDPOINT: GET /health — Cek Status Server

    Ibarat 'tombol intercom' untuk memastikan server masih hidup.
    Digunakan untuk:
    - Memastikan server sudah berjalan setelah deploy
    - Monitoring otomatis oleh Render.com
    - Testing koneksi dari aplikasi PHP sebelum kirim gambar

    Contoh response:
    {
        "status": "ok",
        "mode": "online",
        "model": "tflite",
        "message": "API is running"
    }
    """
    return jsonify({
        "status": "ok",
        "mode": MODE,                                    # "local" atau "online"
        "model": "tflite" if USE_TFLITE else "keras",   # Jenis model yang sedang aktif
        "message": "API is running"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    ENDPOINT: POST /predict — Prediksi Penyakit Daun Padi

    Ini adalah endpoint utama — 'jantung' dari seluruh sistem.
    Menerima gambar dari aplikasi PHP, memprosesnya, dan mengembalikan hasil.

    INPUT  : Form-data dengan field 'image' berisi file gambar (JPG/PNG)
    OUTPUT : JSON berisi label penyakit, confidence, dan probabilitas semua kelas

    ALUR KERJA:
    (1) Validasi: pastikan field 'image' ada di request
    (2) Validasi: pastikan nama file tidak kosong
    (3) Preprocess gambar (resize, normalize, dsb.)
    (4) Jalankan prediksi model AI
    (5) Tentukan kelas dengan probabilitas tertinggi
    (6) Kembalikan hasil sebagai JSON

    Contoh response sukses:
    {
        "label": "Bacterialblight",
        "confidence": 0.9345,
        "probs": {
            "Healthy": 0.0123,
            "Bacterialblight": 0.9345,
            ...
        }
    }
    """
    # (1) Validasi: cek apakah field 'image' ada di request yang dikirim PHP
    if "image" not in request.files:
        return jsonify({"error": "Field 'image' tidak ditemukan."}), 400

    file = request.files["image"]

    # (2) Validasi: cek apakah nama file tidak kosong (antisipasi upload tanpa file)
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    try:
        # (3) Preprocess gambar: resize, konversi warna, normalisasi
        img = preprocess_image(file.read())

        # (4) Jalankan prediksi — otomatis pakai TFLite atau Keras
        preds = predict_model(img)  # Hasilnya: array 5 angka probabilitas [0.02, 0.93, ...]

        # (5) Cari index dengan nilai probabilitas tertinggi → itulah kelas yang diprediksi
        idx        = int(np.argmax(preds))  # Contoh: index 1 = Bacterialblight
        confidence = float(preds[idx])      # Ambil nilai probabilitasnya sebagai confidence

        # (6) Kembalikan hasil sebagai JSON ke aplikasi PHP
        return jsonify({
            "label":      CLASS_NAMES[idx],  # Nama penyakit yang diprediksi (contoh: "Bacterialblight")
            "confidence": confidence,         # Tingkat keyakinan 0.0-1.0 (contoh: 0.9345 = 93.45%)
            "probs": {                        # Probabilitas lengkap semua 5 kelas
                CLASS_NAMES[i]: float(p)
                for i, p in enumerate(preds)
            }
        }), 200

    except Exception as e:
        # Jika terjadi error apapun (gambar rusak, model error, dsb.) → kembalikan pesan error
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================
# JALANKAN SERVER
# ==========================

if __name__ == "__main__":
    # Blok ini HANYA dijalankan jika file dieksekusi langsung (python api_flask.py)
    # Jika dijalankan via Gunicorn (di server production), blok ini DILEWATI
    # karena Gunicorn langsung memanggil objek 'app', bukan menjalankan file ini sebagai script

    # ===============================
    # MODE LOCAL (UNTUK TESTING DI KOMPUTER SENDIRI)
    # ===============================
    if MODE == "local":
        print("[INFO] Running in LOCAL mode")
        app.run(
            host="127.0.0.1",  # Hanya bisa diakses dari komputer sendiri (localhost)
            port=5000,         # Berjalan di port 5000 → http://127.0.0.1:5000
            debug=True         # Mode debug: tampilkan error lengkap & auto-reload saat kode diubah
        )

    # ==================================
    # MODE ONLINE (GUNICORN YANG MENGATUR)
    # Saat di server Render.com, start.sh menjalankan Gunicorn secara langsung
    # Gunicorn mengimpor objek 'app' dari file ini, bukan menjalankan blok if __name__ ini
    # ==================================
    else:
        print("[INFO] Running in ONLINE mode via Gunicorn")
