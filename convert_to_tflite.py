"""
=======================================================
FILE: convert_to_tflite.py
TUJUAN: Mengkonversi model Keras H5 (hasil training di Colab)
        menjadi format TFLite yang jauh lebih ringan,
        agar bisa dijalankan di server gratis seperti Render.com.

Ibarat "mengecilkan ukuran foto" dari resolusi tinggi ke resolusi
yang lebih kecil — kualitas hampir sama, tapi ukuran file jauh lebih kecil.

PILIHAN KONVERSI (PILIH SALAH SATU):
  FP16 (Float 16-bit):
    - Ukuran  : ~17 MB (dari ~100 MB)
    - Akurasi : hampir sama dengan H5 ← SANGAT DIREKOMENDASIKAN
    - Cocok   : untuk Render.com free tier

  INT8 (Integer 8-bit):
    - Ukuran  : ~9 MB (paling kecil)
    - Akurasi : bisa sedikit turun karena presisi lebih rendah
    - Butuh   : dataset contoh untuk proses kalibrasi
    - Cocok   : jika RAM server sangat terbatas

CARA PAKAI DI GOOGLE COLAB:
  1. Pastikan training (train_colab.py) sudah selesai
     dan file best_model.h5 sudah ada di folder model/
  2. Upload atau jalankan file ini di Colab
  3. Tunggu proses konversi selesai (biasanya < 5 menit)
  4. Download file .tflite yang dihasilkan dari folder model/
  5. Upload file tersebut ke repo GitHub app-deteksi_ml/model/

CATATAN PENTING:
  Preprocessing (ukuran gambar, BGR→RGB, normalisasi) di file TFLite
  HARUS IDENTIK dengan yang ada di api_flask.py!
=======================================================
"""

import tensorflow as tf
import numpy as np
import os

# ===========================================================================
# KONFIGURASI PATH
# Semua file model disimpan di subfolder model/ agar lebih rapi.
# Ubah MODEL_DIR jika struktur folder berbeda.
# ===========================================================================

# Tentukan folder root script ini (direktori di mana convert_to_tflite.py berada)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# Folder khusus untuk menyimpan semua file model
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)  # Buat folder jika belum ada

# Path file input: model Keras H5 yang sudah di-training di Colab
MODEL_H5_PATH     = os.path.join(MODEL_DIR, "best_model.h5")

# Path file output FP16: lebih kecil (~17MB), akurasi hampir sama
OUTPUT_FP16_PATH  = os.path.join(MODEL_DIR, "model_fp16.tflite")

# Path file output INT8: paling kecil (~9MB), akurasi bisa sedikit turun
OUTPUT_INT8_PATH  = os.path.join(MODEL_DIR, "model_int8.tflite")

# Ukuran gambar — HARUS sama dengan saat training dan api_flask.py!
IMG_SIZE = (128, 128)

# Path dataset untuk kalibrasi INT8 (ubah sesuai lokasi Anda di Colab)
DATASET_SAMPLE_PATH = "dataset_padi"

# ============================================================
# LANGKAH 1: MUAT MODEL H5
# Baca file model Keras (.h5) dari folder model/
# ============================================================
print(f"[INFO] Memuat model H5 dari: {MODEL_H5_PATH}")
model = tf.keras.models.load_model(MODEL_H5_PATH)
model.summary()  # Tampilkan ringkasan arsitektur model (layer, parameter)
print("[OK] Model H5 berhasil dimuat!\n")

# ============================================================
# LANGKAH 2 (OPSI 1): KONVERSI KE FP16
# "FP16" = Float Point 16-bit. Angka desimal disimpan dengan
# presisi lebih kecil (16-bit alih-alih 32-bit).
# Hasilnya: ukuran file ~2x lebih kecil, akurasi hampir tidak berubah.
# ← INI YANG DIREKOMENDASIKAN untuk Render.com
# ============================================================
print("[INFO] Mengkonversi ke TFLite FP16 (disarankan untuk Render.com)...")

converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)

# Aktifkan optimasi default (mengurangi ukuran model tanpa pengaturan tambahan)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]

# Targetkan tipe data float16 agar bilangan desimal dikompres dari 32-bit → 16-bit
converter_fp16.target_spec.supported_types = [tf.float16]

# Jalankan proses konversi
tflite_fp16 = converter_fp16.convert()

# Simpan hasil ke folder model/
with open(OUTPUT_FP16_PATH, "wb") as f:
    f.write(tflite_fp16)

size_fp16 = len(tflite_fp16) / (1024 * 1024)  # Hitung ukuran dalam Megabyte
print(f"[OK] Selesai! → {OUTPUT_FP16_PATH}")
print(f"     Ukuran   : {size_fp16:.1f} MB\n")


# ============================================================
# LANGKAH 3 (OPSI 2): KONVERSI KE INT8
# "INT8" = Integer 8-bit. Seluruh angka dikonversi dari desimal
# menjadi bilangan bulat (integer) dengan skala tertentu.
# Proses kalibrasi menggunakan sejumlah gambar contoh untuk
# menentukan skala yang tepat agar akurasi tidak banyak turun.
# Hasilnya: file lebih kecil (~9MB), namun proses lebih lambat.
# ============================================================

def representative_dataset():
    """
    Generator gambar contoh untuk proses kalibrasi INT8.

    Model INT8 perlu melihat sejumlah gambar nyata untuk 'belajar'
    rentang nilai yang biasa diproses — proses ini disebut kalibrasi.
    Tanpa kalibrasi, konversi INT8 akan menghasilkan akurasi yang buruk.

    Alur kerja:
    (1) Jelajahi semua subfolder di DATASET_SAMPLE_PATH
    (2) Ambil maksimal 100 gambar (sudah cukup untuk kalibrasi)
    (3) Preproses setiap gambar IDENTIK dengan api_flask.py:
        BGR → RGB → resize 128x128 → normalisasi 0-1 → expand_dims
    (4) Kirimkan ke konverter sebagai generator
    """
    import cv2
    count = 0
    for root, dirs, files in os.walk(DATASET_SAMPLE_PATH):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 100:
                fpath = os.path.join(root, fname)
                img   = cv2.imread(fpath)
                if img is None:
                    continue

                # [PENTING] Konversi BGR → RGB — harus sama dengan api_flask.py!
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)           # Resize ke 128x128
                img = img.astype("float32") / 255.0       # Normalisasi 0-255 → 0.0-1.0
                img = np.expand_dims(img, axis=0)          # Tambah dimensi batch: (128,128,3) → (1,128,128,3)
                yield [img]
                count += 1

print("[INFO] Mengkonversi ke TFLite INT8 (butuh beberapa menit untuk kalibrasi)...")

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations              = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset    = representative_dataset  # Dataset untuk kalibrasi

# Paksa semua operasi dalam model menggunakan INT8
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Tetap gunakan float32 untuk INPUT dan OUTPUT agar preprocessing tidak perlu diubah
# (hanya bagian dalam model yang INT8, bukan input/output-nya)
converter_int8.inference_input_type  = tf.float32  # Input tetap float32 → preprocessing sama dengan FP16
converter_int8.inference_output_type = tf.float32  # Output tetap float32 → probabilitas masih desimal

tflite_int8 = converter_int8.convert()

# Simpan hasil ke folder model/
with open(OUTPUT_INT8_PATH, "wb") as f:
    f.write(tflite_int8)

size_int8 = len(tflite_int8) / (1024 * 1024)
print(f"[OK] Selesai! → {OUTPUT_INT8_PATH}")
print(f"     Ukuran   : {size_int8:.1f} MB\n")

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("=" * 50)
print("RINGKASAN HASIL KONVERSI:")
print(f"  FP16 : {size_fp16:.1f} MB  ← Pilih ini untuk Render.com (akurasi terbaik)")
print(f"  INT8 : {size_int8:.1f} MB  ← Alternatif jika RAM Render.com masih kurang")
print("=" * 50)
print("\nLangkah selanjutnya:")
print("  1. Download model_fp16.tflite dari folder model/ di Colab")
print("  2. Letakkan di folder app-deteksi_ml/model/ di komputer Anda")
print("  3. Commit & push ke GitHub")
print("  4. Render.com akan otomatis deploy ulang menggunakan model baru")
