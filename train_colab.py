"""
=======================================================
TRAINING MODEL DETEKSI PENYAKIT PADI - GOOGLE COLAB
=======================================================
Kelas: Bacterialblight, Blast, Brownspot, Healthy, Tungro
Input: 128x128 RGB | Output: 5 kelas softmax

CARA PAKAI:
  Jalankan sel satu per satu dari atas ke bawah.
  Setiap "# ===== SEL X =====" adalah 1 cell baru di Colab.
=======================================================
"""

# ============================================================
# SEL 1: SETUP KAGGLE & DOWNLOAD DATASET
# ============================================================
import os
import shutil

os.environ['KAGGLE_USERNAME'] = "lulukaulani08"
os.environ['KAGGLE_KEY'] = "KGAT_43cee82e0c48baacc74ea2ebfdb30ba2"

# ----------------------------------------------------------
# STEP 1: Download 4 penyakit dari dataset utama
# ----------------------------------------------------------
os.system("kaggle datasets download -d nirmalsankalana/rice-leaf-disease-image")
os.system("unzip -q rice-leaf-disease-image.zip -d dataset_sementara")

os.makedirs("dataset_padi", exist_ok=True)
os.system("mv dataset_sementara/* dataset_padi/")
os.system("rm -rf dataset_sementara rice-leaf-disease-image.zip")

# ----------------------------------------------------------
# STEP 2: Download kelas Healthy
# Source: https://www.kaggle.com/datasets/dedeikhsandwisaputra/rice-leafs-disease-dataset
# Ambil hanya dari folder TRAIN (bukan validation)
# ----------------------------------------------------------
os.system("kaggle datasets download -d dedeikhsandwisaputra/rice-leafs-disease-dataset")
os.system("unzip -q rice-leafs-disease-dataset.zip -d dataset_healthy_src")

# Cari folder train/healthy secara otomatis (case-insensitive)
healthy_train_src = None
for root, dirs, files in os.walk("dataset_healthy_src"):
    for d in dirs:
        full_path = os.path.join(root, d)
        # Hanya ambil dari path yang mengandung 'train', bukan 'validation'
        if d.lower() == "healthy" and "train" in root.lower():
            healthy_train_src = full_path
            break
    if healthy_train_src:
        break

if healthy_train_src:
    print(f"[INFO] Folder train/healthy ditemukan: {healthy_train_src}")
    os.system(f"mv '{healthy_train_src}' dataset_padi/Healthy")
    print("[OK] Folder Healthy berhasil dipindahkan ke dataset_padi/Healthy")
else:
    # Fallback: tampilkan semua folder untuk debug manual
    print("[ERROR] Folder train/healthy tidak ditemukan. Daftar semua folder:")
    for root, dirs, files in os.walk("dataset_healthy_src"):
        for d in dirs:
            print(f"  {os.path.join(root, d)}")

# Bersihkan file sementara
os.system("rm -rf dataset_healthy_src rice-leafs-disease-dataset.zip")

# ----------------------------------------------------------
# STEP 3: Verifikasi hasil akhir
# ----------------------------------------------------------
print("\n===== Dataset siap di-training =====")
total_all = 0
for kelas in sorted(os.listdir("dataset_padi")):
    path_kelas = f"dataset_padi/{kelas}"
    if os.path.isdir(path_kelas):
        jumlah = len([f for f in os.listdir(path_kelas)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_all += jumlah
        print(f"  {kelas:20s}: {jumlah:>5} gambar")
print(f"  {'TOTAL':20s}: {total_all:>5} gambar")


# ============================================================
# SEL 2: SPLIT DATASET 70% TRAIN / 10% VAL / 20% TEST
# (Split fisik per folder, stratified per kelas)
# ============================================================
import os
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DATASET_SRC = "dataset_padi"

SPLIT_DIRS = {
    'train': 'dataset_train',   # 70%
    'val':   'dataset_val',     # 10%
    'test':  'dataset_test',    # 20%
}

# Bersihkan folder split lama jika ada
for d in SPLIT_DIRS.values():
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)

print("===== Split Dataset 70% / 10% / 20% =====")
print(f"{'Kelas':20s} {'Train':>7} {'Val':>6} {'Test':>6} {'Total':>7}")
print("-" * 50)

for kelas in sorted(os.listdir(DATASET_SRC)):
    src_kelas = os.path.join(DATASET_SRC, kelas)
    if not os.path.isdir(src_kelas):
        continue

    # Kumpulkan semua file gambar
    files = [f for f in os.listdir(src_kelas)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(RANDOM_SEED)

    # Split 1: pisahkan test 20% dulu
    train_val_files, test_files = train_test_split(
        files, test_size=0.20, random_state=RANDOM_SEED
    )
    # Split 2: dari sisa 80%, ambil val 12.5% → hasilnya 10% dari total
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.125, random_state=RANDOM_SEED
    )

    # Salin file ke folder masing-masing
    for subset, file_list in [('train', train_files),
                               ('val',   val_files),
                               ('test',  test_files)]:
        dest = os.path.join(SPLIT_DIRS[subset], kelas)
        os.makedirs(dest, exist_ok=True)
        for fname in file_list:
            shutil.copy2(os.path.join(src_kelas, fname), os.path.join(dest, fname))

    print(f"  {kelas:20s} {len(train_files):>7} {len(val_files):>6} {len(test_files):>6} {len(files):>7}")

print("-" * 50)
print("[OK] Split selesai!\n")


# ============================================================
# SEL 3: TRAINING MODEL
# ============================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, Input, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==============================
# KONFIGURASI
# ==============================
IMG_SIZE   = (128, 128)
BATCH_SIZE = 15      # sesuai skripsi: step/epoch = 4478/15 ≈ 299
EPOCHS     = 100

# Path folder hasil split Sel 2
SPLIT_DIRS = {
    'train': 'dataset_train',
    'val':   'dataset_val',
    'test':  'dataset_test',
}

# ==============================
# URUTAN KELAS (CUSTOM)
# Healthy di index 0, sisanya urut abjad.
# HARUS SAMA PERSIS dengan CLASS_NAMES di api_flask.py!
# ==============================
CLASS_NAMES = [
    "Healthy",          # index 0
    "Bacterialblight",  # index 1
    "Blast",            # index 2
    "Brownspot",        # index 3
    "Tungro",           # index 4
]
NUM_CLASSES = len(CLASS_NAMES)

# ==============================
# GENERATOR — masing-masing dari folder terpisah
# ==============================
# Training: dengan augmentasi
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
# Validasi & Test: hanya rescale, tanpa augmentasi
datagen_eval = ImageDataGenerator(rescale=1./255)

print("[INFO] Memuat Data Training (70%)...")
train_generator = datagen_train.flow_from_directory(
    SPLIT_DIRS['train'],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,   # paksa urutan custom
    shuffle=True
)

print("[INFO] Memuat Data Validasi (10%)...")
val_generator = datagen_eval.flow_from_directory(
    SPLIT_DIRS['val'],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print("[INFO] Memuat Data Testing (20%)...")
test_generator = datagen_eval.flow_from_directory(
    SPLIT_DIRS['test'],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

# Verifikasi urutan kelas
print(f"\n[INFO] Indeks kelas: {train_generator.class_indices}")
assert train_generator.class_indices == {
    'Healthy': 0, 'Bacterialblight': 1, 'Blast': 2, 'Brownspot': 3, 'Tungro': 4
}, "[ERROR] Urutan kelas tidak sesuai!"
print("[OK] Urutan kelas sesuai!")
print(f"  Training : {train_generator.samples} gambar")
print(f"  Validasi : {val_generator.samples} gambar")
print(f"  Testing  : {test_generator.samples} gambar")

# ==============================
# ARSITEKTUR MODEL CNN
# ==============================
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # RGB input

    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    # Block 4
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    # Classifier Head
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ==============================
# MULAI TRAINING
# ==============================
print("[INFO] Memulai Training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

model.save("model_final.h5")
print("[INFO] Training Selesai!")
print("  → best_model.h5  : model terbaik berdasarkan val_accuracy (GUNAKAN INI)")
print("  → model_final.h5 : model di epoch terakhir (backup)")


# ============================================================
# SEL 4: EVALUASI & PLOT GRAFIK + UJI DATA TESTING
# ============================================================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ----- Plot Grafik Training -----
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(acc,     label='Train Accuracy')
axes[0].plot(val_acc, label='Val Accuracy')
axes[0].set_title('Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(loss,     label='Train Loss')
axes[1].plot(val_loss, label='Val Loss')
axes[1].set_title('Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("[INFO] Grafik disimpan sebagai training_history.png")

# ----- Evaluasi pada Data Testing (20%) -----
print("\n===== EVALUASI MODEL TERBAIK PADA DATA TESTING (20%) =====")
best_model = tf.keras.models.load_model("best_model.h5")

test_loss, test_acc = best_model.evaluate(test_generator, verbose=1)
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")

# ----- Classification Report (Precision / Recall / F1) -----
print("\n[INFO] Menghitung Precision, Recall, F1-Score per kelas...")
test_generator.reset()
y_pred_probs = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\n" + classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    digits=4
))

# ----- Confusion Matrix -----
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Data Testing')
plt.ylabel('Label Asli')
plt.xlabel('Label Prediksi')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Confusion matrix disimpan sebagai confusion_matrix.png")


# ============================================================
# SEL 5: KONVERSI KE TFLITE FP16 — UNTUK RENDER.COM
# ============================================================
print("[INFO] Konversi best_model.h5 → model_fp16.tflite...")

model_to_convert = tf.keras.models.load_model("best_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("model_fp16.tflite", "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"[OK] model_fp16.tflite selesai → {size_mb:.1f} MB")
print("\n>>> Download file ini lalu upload ke GitHub repo app-deteksi_ml: <<<")
print("    model_fp16.tflite")
print(f"\n>>> CLASS_NAMES untuk api_flask.py (sudah sesuai): <<<")
print(f"    {CLASS_NAMES}")


# ============================================================
# SEL 6: VERIFIKASI — TEST PREDICT 1 GAMBAR
# ============================================================
import cv2
import glob as glob_module   # alias agar tidak bentrok dengan variable lain

def test_predict_tflite(image_path, class_names):
    """
    Test prediksi dengan TFLite — preprocessing SAMA PERSIS dengan api_flask.py.
    BGR → RGB → resize → normalize → expand_dims
    """
    interp = tf.lite.Interpreter(model_path="model_fp16.tflite")
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    # Preprocessing — harus identik dengan api_flask.py
    img = cv2.imread(image_path)               # OpenCV: BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # → RGB (sama dengan training)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    interp.set_tensor(in_det[0]["index"], img)
    interp.invoke()
    preds = interp.get_tensor(out_det[0]["index"])[0]

    idx  = int(np.argmax(preds))
    conf = float(preds[idx])

    print(f"\nHasil Prediksi: {class_names[idx]} ({conf*100:.1f}%)")
    print("-" * 40)
    for cls, prob in zip(class_names, preds):
        bar = "█" * int(prob * 30)
        print(f"  {cls:20s}: {prob*100:5.1f}%  {bar}")

    return class_names[idx], conf

# Cari gambar sample dari setiap kelas untuk test
print("===== TEST PREDIKSI =====")
for kelas in CLASS_NAMES:
    samples = glob_module.glob(f"dataset_padi/{kelas}/*.jpg")[:1] + \
              glob_module.glob(f"dataset_padi/{kelas}/*.JPG")[:1] + \
              glob_module.glob(f"dataset_padi/{kelas}/*.jpeg")[:1]
    if samples:
        print(f"\n[TEST] Kelas: {kelas} | Gambar: {samples[0]}")
        test_predict_tflite(samples[0], CLASS_NAMES)
    else:
        print(f"[SKIP] Tidak ada gambar sample untuk kelas {kelas}")
