"""
=======================================================
FILE: train_colab.py
TUJUAN: Melatih model CNN untuk mendeteksi 5 jenis kondisi daun padi
        menggunakan Google Colab (komputasi gratis dari Google).

Ibarat 'melatih seorang dokter muda' dari awal:
  - Dokter diberi ribuan foto daun padi beserta diagnosisnya (dataset)
  - Dokter belajar mengenali pola penyakit dari foto-foto tersebut (training)
  - Setelah selesai, kemampuan dokter disimpan ke file H5 (model tersimpan)
  - File H5 kemudian dikompres menjadi TFLite untuk dipakai di server

KELAS YANG DIDETEKSI (5 kelas):
  0. Healthy        → Daun sehat, tidak ada penyakit
  1. Bacterialblight → Hawar Bakteri — bercak cokelat di ujung daun
  2. Blast           → Blas / Busuk Leher — bercak belah ketupat
  3. Brownspot       → Bercak Cokelat — titik-titik cokelat bulat
  4. Tungro          → Tungro (virus) — daun menguning dan menggulung

SPESIFIKASI MODEL:
  Input  : gambar 128 x 128 piksel, 3 saluran warna (RGB)
  Output : 5 angka probabilitas (softmax), total = 1.0
  Contoh output: [0.02, 0.93, 0.02, 0.02, 0.01]
  → Index 1 (Bacterialblight) paling tinggi = penyakit hawar bakteri

CARA PAKAI:
  Buka Google Colab → upload file ini atau copy-paste per sel.
  Jalankan sel satu per satu dari atas ke bawah (jangan loncat-loncat).
  Setiap blok '# ===== SEL X =====' = 1 cell baru di Colab.
=======================================================
"""

# ============================================================
# SEL 1: SETUP KAGGLE & PERSIAPAN DATASET
# ============================================================
# Prioritas sumber data:
#   1. Foto upload manual (hasil dokumentasi lapangan sendiri) — PRIORITAS UTAMA
#   2. Dataset Kaggle — hanya mengisi KEKURANGAN dari target per kelas
#
# Target jumlah data per kelas (sesuai tabel 3.1 skripsi):
#   Bacterialblight : 1535 | Blast    : 1477 | Brownspot : 1696
#   Tungro          : 1329 | Healthy  :  360 | TOTAL     : 6397
# ============================================================
import os
import shutil
import random

# ==============================
# KREDENSIAL KAGGLE API
# Dibutuhkan agar Colab bisa mengunduh dataset dari Kaggle secara otomatis.
# Ganti dengan kredensial akun Kaggle Anda jika token di bawah sudah kedaluwarsa.
# Cara mendapatkan API key: Kaggle → Account → Create New Token → kaggle.json
# ==============================
os.environ['KAGGLE_USERNAME'] = "lulukaulani08"
os.environ['KAGGLE_KEY'] = "KGAT_43cee82e0c48baacc74ea2ebfdb30ba2"

# ==============================
# TARGET JUMLAH DATA PER KELAS (Tabel 3.1 Skripsi)
# Kunci = nama folder kelas, Nilai = total data yang dibutuhkan
# ==============================
TARGET_PER_KELAS = {
    "Bacterialblight": 1535,
    "Blast":           1477,
    "Brownspot":       1696,
    "Tungro":          1329,
    "Healthy":          360,
}
CLASSES      = list(TARGET_PER_KELAS.keys())
UPLOAD_DIR   = "uploads"       # Folder utama tempat Anda menaruh foto manual
DATASET_DEST = "dataset_padi"  # Folder dataset akhir yang akan digunakan training
EKSTENSI_VALID = ('.jpg', '.jpeg', '.png')

# ============================================================
# STEP 1: BUAT FOLDER UPLOAD & TAMPILKAN INSTRUKSI
# ============================================================
# Foto dari upload manual DIPRIORITASKAN terlebih dahulu.
# Kaggle hanya akan digunakan untuk mengisi KEKURANGAN dari target.
# ============================================================
for kelas in CLASSES:
    os.makedirs(os.path.join(UPLOAD_DIR, kelas), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DEST, kelas), exist_ok=True)

print("\n" + "="*60)
print("  📸  SILAKAN UPLOAD FOTO DOKUMENTASI ANDA SEKARANG  📸")
print("="*60)
print("""
  Foto Anda akan DIPRIORITASKAN sebagai data utama.
  Kaggle hanya mengisi sisa kekurangan dari target.

  Langkah-langkah:
  1. Klik ikon 📁 (Files) di panel KIRI Colab
  2. Buka folder  uploads/<nama_kelas>/  sesuai jenis foto:
       📁 uploads/Healthy/          → daun sehat           (target total: 360)
       📁 uploads/Bacterialblight/  → hawar bakteri        (target total: 1535)
       📁 uploads/Blast/            → blas / busuk leher   (target total: 1477)
       📁 uploads/Brownspot/        → bercak cokelat       (target total: 1696)
       📁 uploads/Tungro/           → tungro (menguning)   (target total: 1329)
  3. Klik kanan pada folder → "Upload" → pilih foto (JPG/PNG)
  4. Tunggu hingga semua foto selesai terupload
  5. Tekan ENTER di bawah untuk melanjutkan
""")
print("="*60)
input("  ▶  Tekan ENTER setelah selesai upload (atau ENTER langsung jika tidak ada)...")
print()

# ============================================================
# STEP 2: HITUNG FOTO UPLOAD & SALIN KE dataset_padi/
# ============================================================
# Foto upload manual disalin terlebih dahulu ke dataset_padi/<kelas>/.
# Sistem mencatat jumlah per kelas untuk menentukan sisa kebutuhan Kaggle.
# ============================================================
print("[INFO] Memproses foto upload...")

jumlah_upload = {}   # Menyimpan jumlah foto upload per kelas
for kelas in CLASSES:
    src_folder = os.path.join(UPLOAD_DIR, kelas)
    dst_folder = os.path.join(DATASET_DEST, kelas)

    foto_list = sorted([
        f for f in os.listdir(src_folder)
        if f.lower().endswith(EKSTENSI_VALID)
    ])

    copied = 0
    for fname in foto_list:
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)
        # Skip jika sudah ada — hindari tumpuk duplikat saat re-run
        if os.path.exists(dst_path):
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1

    jumlah_upload[kelas] = len(foto_list)

# ============================================================
# STEP 3: DOWNLOAD DATASET KAGGLE (HANYA UNTUK MENGISI KEKURANGAN)
# ============================================================
# Hitung sisa dari isi dataset_padi/ yang SUDAH ADA (bukan hanya upload).
# Re-run aman: jika target sudah terpenuhi → skip download & salin.
# ============================================================

def _hitung_gambar(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(EKSTENSI_VALID)])

jumlah_existing = {
    k: _hitung_gambar(os.path.join(DATASET_DEST, k)) for k in CLASSES
}
kelas_butuh_kaggle = {k: max(0, TARGET_PER_KELAS[k] - jumlah_existing[k]) for k in CLASSES}
total_butuh        = sum(kelas_butuh_kaggle.values())

if total_butuh == 0:
    print("[INFO] Dataset sudah lengkap — skip download Kaggle.")
else:
    print("[INFO] Mengunduh dataset Kaggle...")

    # ----------------------------------------------------------
    # STEP 3a: Download 4 kelas penyakit dari dataset utama
    # Sumber: https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image
    # Berisi: Bacterialblight, Blast, Brownspot, Tungro
    # ----------------------------------------------------------
    butuh_penyakit = sum(kelas_butuh_kaggle[k] for k in ["Bacterialblight", "Blast", "Brownspot", "Tungro"])
    if butuh_penyakit > 0:
        zip_penyakit = "rice-leaf-disease-image.zip"
        if not os.path.isdir("dataset_kaggle_src"):
            if not os.path.isfile(zip_penyakit):
                os.system("kaggle datasets download -d nirmalsankalana/rice-leaf-disease-image")
            os.system(f"unzip -q {zip_penyakit} -d dataset_kaggle_src")

        # Salin dari Kaggle hanya sebanyak kekurangan per kelas
        for kelas in ["Bacterialblight", "Blast", "Brownspot", "Tungro"]:
            sisa = kelas_butuh_kaggle[kelas]
            if sisa == 0:
                continue

            # Cari folder kelas di dalam hasil ekstrak Kaggle (case-insensitive)
            src_kelas = None
            for root, dirs, _ in os.walk("dataset_kaggle_src"):
                for d in dirs:
                    if d.lower() == kelas.lower():
                        src_kelas = os.path.join(root, d)
                        break
                if src_kelas:
                    break

            if not src_kelas:
                continue

            dst_folder = os.path.join(DATASET_DEST, kelas)
            semua_foto = sorted([
                f for f in os.listdir(src_kelas)
                if f.lower().endswith(EKSTENSI_VALID)
            ])
            random.seed(42)
            random.shuffle(semua_foto)

            copied = 0
            for fname in semua_foto:
                if copied >= sisa:
                    break
                src_path = os.path.join(src_kelas, fname)
                dst_path = os.path.join(dst_folder, fname)
                # Skip jika sudah ada — jangan buat _k duplikat
                if os.path.exists(dst_path):
                    continue
                shutil.copy2(src_path, dst_path)
                copied += 1

        os.system("rm -rf dataset_kaggle_src rice-leaf-disease-image.zip")

    # ----------------------------------------------------------
    # STEP 3b: Download kelas Healthy dari dataset kedua
    # Source: https://www.kaggle.com/datasets/dedeikhsandwisaputra/rice-leafs-disease-dataset
    # Ambil hanya dari folder TRAIN (bukan validation)
    # ----------------------------------------------------------
    sisa_healthy = kelas_butuh_kaggle["Healthy"]
    if sisa_healthy == 0:
        pass  # sudah terpenuhi
    else:
        zip_healthy = "rice-leafs-disease-dataset.zip"
        if not os.path.isdir("dataset_healthy_src"):
            if not os.path.isfile(zip_healthy):
                os.system("kaggle datasets download -d dedeikhsandwisaputra/rice-leafs-disease-dataset")
            os.system(f"unzip -q {zip_healthy} -d dataset_healthy_src")

        # Cari folder train/healthy secara otomatis (case-insensitive)
        healthy_train_src = None
        for root, dirs, files in os.walk("dataset_healthy_src"):
            for d in dirs:
                full_path = os.path.join(root, d)
                if d.lower() == "healthy" and "train" in root.lower():
                    healthy_train_src = full_path
                    break
            if healthy_train_src:
                break

        if healthy_train_src:
            dst_folder = os.path.join(DATASET_DEST, "Healthy")
            semua_foto = sorted([
                f for f in os.listdir(healthy_train_src)
                if f.lower().endswith(EKSTENSI_VALID)
            ])
            random.seed(42)
            random.shuffle(semua_foto)

            copied = 0
            for fname in semua_foto:
                if copied >= sisa_healthy:
                    break
                src_path = os.path.join(healthy_train_src, fname)
                dst_path = os.path.join(dst_folder, fname)
                if os.path.exists(dst_path):
                    continue
                shutil.copy2(src_path, dst_path)
                copied += 1
        else:
            print("[ERROR] Folder Healthy dari Kaggle tidak ditemukan.")

        os.system("rm -rf dataset_healthy_src rice-leafs-disease-dataset.zip")

# ----------------------------------------------------------
# STEP 4: Verifikasi hasil akhir — pastikan semua kelas lengkap
# Sistem akan menghitung jumlah gambar per kelas dan menampilkannya
# ----------------------------------------------------------
print("\n===== Dataset siap di-training =====")
total_all = 0
for kelas in CLASSES:
    path_kelas = os.path.join(DATASET_DEST, kelas)
    jumlah = len([f for f in os.listdir(path_kelas)
                  if f.lower().endswith(EKSTENSI_VALID)]) if os.path.isdir(path_kelas) else 0
    total_all += jumlah
    print(f"  {kelas:20s}: {jumlah:>5} gambar")
print(f"  {'TOTAL':20s}: {total_all:>5} gambar")
print("[OK] Dataset selesai dipersiapkan!")


# ============================================================
# SEL 2: SPLIT DATASET 70% TRAIN / 10% VAL / 20% TEST
# ============================================================
# Ibarat memisahkan soal ujian menjadi tiga kelompok:
#   - 70% TRAIN  → soal yang dikerjakan saat belajar (model melihat ini)
#   - 10% VAL    → soal latihan untuk cek apakah sudah belajar dengan benar
#   - 20% TEST   → soal ujian akhir untuk menilai kemampuan sesungguhnya
#
# Penting: setiap kelas dipisahkan secara merata (stratified split)
# agar model tidak belajar lebih banyak dari satu jenis penyakit saja.
# ============================================================
import os
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42  # Angka acak tetap agar split selalu sama jika diulang
DATASET_SRC = "dataset_padi"  # Folder sumber dataset yang sudah lengkap 5 kelas

# Nama folder output untuk masing-masing subset
SPLIT_DIRS = {
    'train': 'dataset_train',   # 70% dari total
    'val':   'dataset_val',     # 10% dari total
    'test':  'dataset_test',    # 20% dari total
}

# Bersihkan folder split lama jika ada (untuk memastikan tidak ada data sisa)
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

    # Kumpulkan semua file gambar dalam kelas ini
    files = [f for f in os.listdir(src_kelas)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(RANDOM_SEED)

    # Split dalam 2 tahap:
    # Tahap 1: pisahkan test (20%) dari semua data
    train_val_files, test_files = train_test_split(
        files, test_size=0.20, random_state=RANDOM_SEED
    )
    # Tahap 2: dari sisa 80%, pisahkan val (12.5% dari 80% = 10% dari total)
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.125, random_state=RANDOM_SEED
    )

    # Salin (bukan memindahkan) file ke folder masing-masing
    # Agar dataset asli di dataset_padi/ tidak rusak
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
# SEL 3: TRAINING MODEL CNN (Convolutional Neural Network)
# ============================================================
# Ini adalah inti dari proses machine learning.
# Model akan 'belajar' mengenali pola visual penyakit daun padi
# dengan melihat ribuan foto berlabel dari folder dataset_train/.
#
# CNN bekerja seperti mata manusia yang terlatih:
#   Layer Conv2D   → mendeteksi garis, warna, tekstur pada gambar
#   MaxPooling2D   → merangkum informasi penting, buang detail tidak perlu
#   Flatten+Dense  → mengambil keputusan akhir berdasarkan pola yang ditemukan
#   Softmax        → mengubah keputusan menjadi probabilitas 0-1 untuk 5 kelas
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
# KONFIGURASI HYPERPARAMETER
# ==============================
IMG_SIZE   = (128, 128)  # Ukuran input gambar — HARUS sama dengan api_flask.py!
BATCH_SIZE = 15           # Jumlah foto yang diproses sekaligus per langkah
                          # (sesuai skripsi: step/epoch = 4478/15 ≈ 299)
EPOCHS     = 100          # Maksimum iterasi belajar (bisa berhenti lebih awal via EarlyStopping)

# ==============================
# FOLDER DATASET
# ==============================
# Path folder hasil split dari Sel 2
SPLIT_DIRS = {
    'train': 'dataset_train',
    'val':   'dataset_val',
    'test':  'dataset_test',
}

# ==============================
# URUTAN KELAS (WAJIB SAMA DENGAN api_flask.py!)
# Keras secara default mengurutkan nama kelas secara alfabetis.
# Kita paksa urutan custom agar Healthy tetap di index 0.
# PERINGATAN: Jangan ubah urutan ini! Harus persis sama
# dengan variabel CLASS_NAMES di api_flask.py!
# ==============================
CLASS_NAMES = [
    "Healthy",          # index 0 — Daun sehat
    "Bacterialblight",  # index 1 — Hawar bakteri
    "Blast",            # index 2 — Blas/busuk leher
    "Brownspot",        # index 3 — Bercak cokelat
    "Tungro",           # index 4 — Tungro (virus)
]
NUM_CLASSES = len(CLASS_NAMES)  # = 5

# ==============================
# DATA GENERATOR (AUGMENTASI)
# ==============================
# Data Generator adalah 'mesin persiapan foto' sebelum dimasukkan ke model.
# Untuk data training: selain menyiapkan foto, generator juga memvariasikan
# foto agar model lebih tahan terhadap berbagai kondisi pengambilan gambar.
# Untuk data validasi & testing: hanya normalisasi, tanpa variasi.
#
# AUGMENTASI yang diterapkan saat training:
#   rotation_range=25       → putar foto hingga 25 derajat (kiri/kanan)
#   width_shift_range=0.2   → geser horizontal hingga 20% lebar foto
#   height_shift_range=0.2  → geser vertikal hingga 20% tinggi foto
#   zoom_range=0.2          → perbesar/perkecil gambar hingga 20%
#   horizontal_flip=True    → cerminkan foto secara horizontal
#   brightness_range=[0.8, 1.2] → variasikan kecerahan (80%-120%)
#
# rescale=1./255 pada semua generator:
#   Mengubah nilai piksel dari rentang 0-255 menjadi 0.0-1.0
#   (HARUS IDENTIK dengan normalisasi di api_flask.py!)
# ==============================
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
# Validasi & Test: HANYA rescale, tanpa augmentasi
# (penting! model harus dievaluasi dengan foto 'asli', bukan yang sudah divariasikan)
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
# Model ini tersusun dari 4 blok konvolusi (Conv2D) + 1 kepala klasifikasi.
# Setiap blok secara bertahap mengekstrak fitur yang semakin abstrak:
#   Block 1 (32 filter)  → mendeteksi tepi, warna, tekstur dasar
#   Block 2 (64 filter)  → mendeteksi pola lebih kompleks (bercak, garis)
#   Block 3 (128 filter) → mendeteksi bentuk lokal penyakit
#   Block 4 (256 filter) → mendeteksi pola global (distribusi penyakit di daun)
#
# BatchNormalization → menstabilkan proses belajar agar lebih cepat konvergen
# Dropout           → mencegah overfitting (model terlalu hafal, bukan belajar)
# Softmax (output)  → mengubah nilai akhir menjadi 5 probabilitas yang total = 1.0
# ==============================
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Input: gambar 128x128 RGB (3 saluran warna)

    # Block 1 — filter kecil (32), tangkap fitur dasar
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 2 — filter lebih banyak (64), deteksi pola lebih rumit
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),  # Matikan 30% neuron secara acak agar tidak terlalu bergantung pada satu pola

    # Block 3 — filter 128, fitur semakin spesifik
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),  # Dropout lebih tinggi karena fitur semakin kompleks

    # Block 4 — filter 256, fitur paling abstrak dan high-level
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    # Kepala Klasifikasi — ubah peta fitur 2D menjadi keputusan final
    Flatten(),         # Ratakan peta fitur 3D menjadi 1D
    Dense(512, activation='relu'),  # Layer tersembunyi besar (512 neuron)
    BatchNormalization(),
    Dropout(0.5),      # Dropout 50% — regularisasi kuat sebelum output
    Dense(NUM_CLASSES, activation='softmax')  # Output: 5 probabilitas (total=1.0)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam: optimizer adaptif yang umum dipakai
    loss='categorical_crossentropy',       # Loss untuk klasifikasi multi-kelas
    metrics=['accuracy']                   # Metrik yang ditampilkan saat training
)

model.summary()  # Tampilkan ringkasan arsitektur model

# ==============================
# CALLBACKS (PENGATUR OTOMATIS SAAT TRAINING)
# ==============================
# Callback ibarat 'asisten yang mengawasi proses belajar' dan
# mengambil tindakan tertentu berdasarkan kondisi yang terjadi.
# ==============================

early_stop = EarlyStopping(
    monitor='val_loss',       # Pantau nilai loss di data validasi
    patience=15,              # Berhenti jika val_loss tidak membaik selama 15 epoch berturut-turut
    restore_best_weights=True, # Kembalikan bobot terbaik saat berhenti (bukan bobot terakhir)
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/best_model.h5",      # Simpan model terbaik ke folder model/
    monitor='val_accuracy',     # Pantau akurasi validasi
    save_best_only=True,        # Hanya simpan jika ini adalah akurasi terbaik sejauh ini
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Pantau nilai loss di data validasi
    factor=0.5,            # Kurangi learning rate menjadi 50% saat stuck
    patience=5,            # Tunggu 5 epoch sebelum mengurangi learning rate
    min_lr=1e-6,           # Batas minimum learning rate (jangan terlalu kecil)
    verbose=1
)

# ==============================
# MULAI TRAINING
# ==============================
print("[INFO] Memulai Training...")
print(f"  Total epoch   : {EPOCHS}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"  Ukuran gambar : {IMG_SIZE}")
print("  EarlyStopping akan otomatis berhenti jika val_loss tidak membaik 15 epoch")
print("  ModelCheckpoint otomatis menyimpan model terbaik ke model/best_model.h5")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Simpan model epoch terakhir sebagai backup (mungkin berbeda dari best_model)
model.save("model/model_final.h5")
print("[INFO] Training Selesai!")
print("  → model/best_model.h5   : model TERBAIK berdasarkan val_accuracy (GUNAKAN INI untuk konversi TFLite)")
print("  → model/model_final.h5  : model di epoch terakhir (backup saja)")


# ============================================================
# SEL 4: EVALUASI & VISUALISASI GRAFIK + UJI DATA TESTING
# ============================================================
# Setelah training selesai, kita perlu mengevaluasi seberapa baik
# model belajar menggunakan DATA TESTING (20% yang belum pernah dilihat model).
#
# Visualisasi yang dihasilkan:
# 1. Grafik Akurasi per Epoch (training_history.png)
#    → Menunjukkan apakah model belajar dengan benar (naik terus) atau overfitting
# 2. Confusion Matrix (confusion_matrix.png)
#    → Menunjukkan di mana model sering salah (kolom/baris yang besar = sering keliru)
# 3. Classification Report (di console)
#    → Precision, Recall, F1-Score per kelas
# ============================================================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ----- PLOT GRAFIK TRAINING -----
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafik kiri: Akurasi
axes[0].plot(acc,     label='Train Accuracy')  # Akurasi di data training
axes[0].plot(val_acc, label='Val Accuracy')    # Akurasi di data validasi
axes[0].set_title('Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Grafik kanan: Loss (semakin kecil semakin bagus)
axes[1].plot(loss,     label='Train Loss')
axes[1].plot(val_loss, label='Val Loss')
axes[1].set_title('Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)  # Simpan grafik sebagai file gambar
plt.show()
print("[INFO] Grafik training disimpan sebagai training_history.png")

# ----- EVALUASI PADA DATA TESTING (yang belum pernah dilihat model) -----
print("\n===== EVALUASI MODEL TERBAIK PADA DATA TESTING (20%) =====")
best_model = tf.keras.models.load_model("model/best_model.h5")  # Muat model terbaik dari folder model/

test_loss, test_acc = best_model.evaluate(test_generator, verbose=1)
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")

# ----- CLASSIFICATION REPORT (Precision / Recall / F1) -----
# Precision  → dari semua prediksi 'Blast', berapa % yang benar-benar Blast?
# Recall     → dari semua foto Blast yang ada, berapa % yang berhasil dideteksi?
# F1-Score   → rata-rata harmonis antara Precision dan Recall
print("\n[INFO] Menghitung Precision, Recall, F1-Score per kelas...")
test_generator.reset()
y_pred_probs = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)  # Pilih index dengan probabilitas tertinggi
y_true = test_generator.classes            # Label asli dari dataset testing

print("\n" + classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    digits=4
))

# ----- CONFUSION MATRIX -----
# Matriks ini menunjukkan:
#   - Diagonal utama (kiri-atas ke kanan-bawah) = prediksi BENAR
#   - Di luar diagonal = prediksi SALAH (baris = asli, kolom = prediksi)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix \u2014 Data Testing')
plt.ylabel('Label Asli')       # Baris = label yang sebenarnya
plt.xlabel('Label Prediksi')   # Kolom = yang diprediksi model
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Confusion matrix disimpan sebagai confusion_matrix.png")


# ============================================================
# SEL 5: KONVERSI KE TFLITE FP16 — UNTUK RENDER.COM
# ============================================================
# Setelah model selesai dilatih, kita perlu mengkonversinya ke format
# yang lebih ringan agar bisa berjalan di server gratis Render.com.
#
# FP16 (Float 16-bit) = ukuran file dikurangi ~6x (dari ~100MB → ~17MB)
# dengan cara mengkompres bilangan desimal dari 32-bit menjadi 16-bit.
# Akurasi hampir tidak berubah karena perbedaan presisi sangat kecil.
# ============================================================
print("[INFO] Konversi best_model.h5 → model_fp16.tflite...")

# Muat ulang model terbaik dari folder model/
model_to_convert = tf.keras.models.load_model("model/best_model.h5")

# Setup konverter
converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
converter.optimizations = [tf.lite.Optimize.DEFAULT]          # Aktifkan optimasi otomatis
converter.target_spec.supported_types = [tf.float16]           # Kompres ke presisi 16-bit

# Jalankan konversi
tflite_model = converter.convert()

# Simpan ke folder model/ agar rapi bersama file H5
with open("model/model_fp16.tflite", "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"[OK] model_fp16.tflite selesai → {size_mb:.1f} MB")
print("\n>>> Download file ini lalu upload ke GitHub repo app-deteksi_ml/model/: <<<")
print("    model/model_fp16.tflite")
print(f"\n>>> CLASS_NAMES untuk api_flask.py (sudah sesuai): <<<")
print(f"    {CLASS_NAMES}")


# ============================================================
# SEL 6: VERIFIKASI — TEST PREDIKSI 1 GAMBAR PER KELAS
# ============================================================
# Sebelum deploy, kita verifikasi bahwa model TFLite yang baru
# dibuat memberikan prediksi yang masuk akal.
# Script ini mengambil satu gambar dari setiap kelas di dataset_padi/
# dan menjalankan prediksi persis seperti yang dilakukan api_flask.py.
#
# PENTING: Preprocessing di sini HARUS IDENTIK dengan api_flask.py!
# (BGR→RGB, resize 128x128, normalisasi 0-1, expand_dims)
# ============================================================
import cv2
import glob as glob_module   # Alias agar tidak bentrok dengan variabel lokal bernama 'glob'

def test_predict_tflite(image_path, class_names):
    """
    Menjalankan prediksi satu gambar menggunakan model TFLite.
    Preprocessing HARUS IDENTIK dengan fungsi preprocess_image() di api_flask.py.

    Alur:
    (1) Baca gambar menggunakan OpenCV (format BGR)
    (2) Konversi BGR → RGB (sama dengan training menggunakan ImageDataGenerator)
    (3) Resize ke 128x128 piksel
    (4) Normalisasi 0-255 → 0.0-1.0
    (5) Tambah dimensi batch: (128,128,3) → (1,128,128,3)
    (6) Masukkan ke model TFLite dan ambil hasil prediksi
    (7) Tampilkan hasilnya beserta bar probabilitas
    """
    # Muat model TFLite dari folder model/
    interp = tf.lite.Interpreter(model_path="model/model_fp16.tflite")
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    # (1)-(5) Preprocessing — harus identik dengan api_flask.py
    img = cv2.imread(image_path)               # (1) Baca: OpenCV format BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (2) Konversi → RGB
    img = cv2.resize(img, (128, 128))          # (3) Resize
    img = img.astype("float32") / 255.0        # (4) Normalisasi
    img = np.expand_dims(img, axis=0)          # (5) Tambah dimensi batch

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
