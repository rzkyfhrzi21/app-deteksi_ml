"""
=======================================================
KONVERSI MODEL H5 → TFLITE
=======================================================
Pilihan:
  - FP16 : ~29 MB, akurasi hampir sama dengan H5 (DIREKOMENDASIKAN)
  - INT8  : ~15 MB, akurasi sedikit lebih rendah, paling ringan

Cara pakai di Colab:
  1. Upload model H5 hasil training
  2. Jalankan sel yang diinginkan (FP16 atau INT8)
  3. Download file .tflite yang dihasilkan
  4. Upload ke repo app-deteksi_ml di GitHub/Render
=======================================================
"""

import tensorflow as tf
import numpy as np

# ===========================
# LOAD MODEL H5
# ===========================
print("[INFO] Loading model H5...")
model = tf.keras.models.load_model("best_model.h5")
model.summary()

# ===========================
# OPSI 1: FP16 QUANTIZATION
# Ukuran ~29 MB | Akurasi ≈ sama dengan H5 ← REKOMENDASIKAN INI
# ===========================
print("\n[INFO] Konversi ke TFLite FP16...")
converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]

tflite_fp16 = converter_fp16.convert()

with open("model_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)

size_fp16 = len(tflite_fp16) / (1024 * 1024)
print(f"[OK] model_fp16.tflite selesai → {size_fp16:.1f} MB")


# ===========================
# OPSI 2: INT8 QUANTIZATION (FULL)
# Ukuran ~15 MB | Akurasi bisa sedikit turun
# Butuh dataset contoh (representative dataset) untuk kalibrasi
# ===========================
# Siapkan beberapa gambar contoh untuk kalibrasi INT8
# (Ganti path sesuai folder dataset Anda di Colab)

DATASET_SAMPLE_PATH = "dataset_padi"  # ganti jika perlu
IMG_SIZE = (128, 128)

def representative_dataset():
    """Generator gambar sample untuk kalibrasi INT8"""
    import os, cv2
    count = 0
    for root, dirs, files in os.walk(DATASET_SAMPLE_PATH):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 100:
                fpath = os.path.join(root, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                # [PENTING] Konversi BGR → RGB agar sama dengan training
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                yield [img]
                count += 1

print("\n[INFO] Konversi ke TFLite INT8 (butuh beberapa menit)...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.float32   # tetap float32 agar preprocessing tidak perlu diubah
converter_int8.inference_output_type = tf.float32  # tetap float32 untuk output probabilitas

tflite_int8 = converter_int8.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_int8)

size_int8 = len(tflite_int8) / (1024 * 1024)
print(f"[OK] model_int8.tflite selesai → {size_int8:.1f} MB")

print("\n=== Ringkasan ===")
print(f"  FP16 : {size_fp16:.1f} MB  ← Pilih ini untuk Render.com (akurasi terbaik)")
print(f"  INT8 : {size_int8:.1f} MB  ← Alternatif jika RAM Render masih kurang")
