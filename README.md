# 🤖 app-deteksi_ml — Model AI & Server Flask Deteksi Penyakit Daun Padi

Ini adalah folder **mesin kecerdasan buatan (AI)** dari sistem deteksi.
Berisi model CNN yang sudah dilatih, server Flask sebagai jembatan API,
dan semua script untuk melatih ulang maupun mengkonversi model.

---

## ✨ Fitur Utama: Deteksi Penyakit Daun Padi dengan CNN

Model **Convolutional Neural Network (CNN)** di folder ini mampu menganalisis foto daun padi dan menentukan jenis penyakitnya secara otomatis — seperti seorang ahli pertanian yang memeriksa daun secara visual, namun dilakukan oleh komputer.

### 🔬 Cara Kerja Model CNN

```
Foto daun padi (JPG/PNG)
        ↓
[1] Preprocess: BGR → RGB → resize 128×128 → normalisasi 0.0–1.0
        ↓
[2] Model CNN (4 blok konvolusi):
    Block 1 (32 filter)  → deteksi tepi, warna, tekstur dasar
    Block 2 (64 filter)  → deteksi pola bercak, garis
    Block 3 (128 filter) → deteksi bentuk lokal penyakit
    Block 4 (256 filter) → deteksi distribusi penyakit di daun
        ↓
[3] Dense + Softmax → 5 angka probabilitas (total = 1.0)
    Contoh: [0.02, 0.93, 0.02, 0.02, 0.01]
        ↓
[4] Index angka tertinggi = kelas penyakit yang diprediksi
    0.93 di index 1 → "Bacterialblight" (93% yakin)
```

### 🦠 Kelas Penyakit yang Dideteksi

| Index | Nama Folder Dataset | Tampilan di Aplikasi | Keterangan |
|-------|---------------------|----------------------|------------|
| 0 | `Healthy` | Healthy (Daun Sehat) | Daun normal, tidak ada penyakit |
| 1 | `Bacterialblight` | Bacterial Blight | Hawar bakteri — bercak cokelat di tepi/ujung daun |
| 2 | `Blast` | Blast | Blas/busuk leher — bercak belah ketupat pada helai daun |
| 3 | `Brownspot` | Brown Spot | Bercak cokelat — titik-titik bulat cokelat tersebar |
| 4 | `Tungro` | Tungro | Virus tungro — daun menguning, menggulung, kerdil |

### 📁 File Kunci Fitur Deteksi

| File | Peran |
|------|-------|
| `api_flask.py` | **🔑 Server API utama** — menerima gambar dari PHP, preprocess, prediksi, kembalikan JSON |
| `model/model_fp16.tflite` | **🧠 Model CNN** — file model terkompresi yang dijalankan saat prediksi |
| `model/best_model.h5` | **💾 Backup model** — model Keras asli (lebih besar, untuk konversi ulang) |
| `train_colab.py` | **🏋️ Script training** — melatih model CNN dari awal di Google Colab |
| `convert_to_tflite.py` | **🔄 Konverter** — mengubah model H5 → TFLite agar lebih ringan |

---

## 📂 Struktur Folder Lengkap

```
app-deteksi_ml/
│
├── api_flask.py               ← ⭐ SERVER API UTAMA (Flask)
│                                 Endpoint GET /health dan POST /predict
│                                 Dijalankan via Gunicorn di Render.com
│
├── model/                     ← 🧠 Folder semua file model AI
│   ├── model_fp16.tflite      ← Model UTAMA — TFLite FP16, ~17 MB
│   │                             Inilah yang dipakai saat prediksi di server
│   ├── best_model.h5          ← Model Keras asli — ~100 MB (backup)
│   │                             Digunakan untuk konversi ulang ke TFLite
│   └── model_final.h5         ← Model Keras epoch terakhir — ~100 MB (backup)
│                                 Simpan sebagai cadangan jika best_model.h5 perlu diuji
│
├── train_colab.py             ← 🏋️ SCRIPT TRAINING (Google Colab)
│                                 6 sel berurutan: download dataset → split → train
│                                 → evaluasi → konversi TFLite → verifikasi
│
├── convert_to_tflite.py       ← 🔄 SCRIPT KONVERSI H5 → TFLite
│                                 Dua opsi: FP16 (~17MB) atau INT8 (~9MB)
│                                 Jalankan di Colab setelah training selesai
│
├── start.sh                   ← Script Bash untuk menjalankan Gunicorn di Render.com
│                                 Isi: gunicorn api_flask:app --workers 1 ...
│
├── requirements.txt           ← Daftar library Python yang dibutuhkan
│                                 (flask, tensorflow, opencv-python, numpy, gunicorn)
│
├── dataset/                   ← Folder referensi dataset lokal
│   ├── DATASET PENYAKIT PADI/ ← Dataset penyakit (Bacterialblight, Blast, Brownspot, Tungro)
│   └── RiceLeafsDisease/      ← Dataset kelas Healthy
│
├── referensi/                 ← Materi & kode referensi untuk skripsi
│   ├── *.ipynb                ← Notebook Jupyter percobaan/riset
│   ├── train_local.py         ← Versi training untuk dijalankan di komputer lokal (bukan Colab)
│   └── requirements.txt       ← Requirements khusus untuk training lokal
│
├── confusion_matrix.png       ← Gambar confusion matrix hasil evaluasi model
├── training_history.png       ← Grafik akurasi & loss per epoch saat training
│
└── README.md                  ← File ini
```

---

## 🚀 Cara Menjalankan Server Flask Secara Lokal

```bash
# 1. Pastikan Python 3.10+ sudah terinstall
# 2. Install dependensi
pip install -r requirements.txt

# 3. Set mode ke local agar debug mode aktif
set FLASK_MODE=local      # Windows CMD
$env:FLASK_MODE="local"   # Windows PowerShell
export FLASK_MODE=local   # Linux/Mac

# 4. Jalankan server
python api_flask.py

# Server berjalan di: http://127.0.0.1:5000
# Cek status: http://127.0.0.1:5000/health
```

---

## ☁️ Deployment di Render.com

Server Flask di-deploy di **Render.com** (free tier) sebagai Web Service Python.

- **URL Produksi** : `https://app-deteksi.onrender.com`
- **Health Check** : `https://app-deteksi.onrender.com/health`
- **Predict Endpoint**: `https://app-deteksi.onrender.com/predict`
- **Start Command** : `sh start.sh` (menjalankan Gunicorn)

> ⚠️ **Catatan Free Tier**: Server Render.com gratis akan "tidur" (spin down) setelah
> 15 menit tidak ada request. Request pertama setelah tidur butuh waktu **30-60 detik**
> untuk "bangun" kembali. Gunakan tombol **"Tes Koneksi"** di aplikasi untuk membangunkan
> server sebelum melakukan deteksi.

---

## 🔄 Alur Kerja Lengkap: Dari Training hingga Produksi

```
[Google Colab]
      │
      ├── train_colab.py (Sel 1-3)
      │   Download dataset Kaggle → Split 70/10/20% → Training CNN
      │   Output: model/best_model.h5
      │
      ├── train_colab.py (Sel 4)
      │   Evaluasi: Accuracy, Precision, Recall, F1, Confusion Matrix
      │
      └── train_colab.py (Sel 5) ATAU convert_to_tflite.py
          Konversi H5 → TFLite FP16
          Output: model/model_fp16.tflite

[Download & Upload ke GitHub]
      model/model_fp16.tflite → push ke repo

[Render.com — Auto Deploy]
      start.sh → gunicorn api_flask:app
      api_flask.py membaca model/model_fp16.tflite

[Aplikasi PHP — app-deteksi]
      function_deteksi.php → cURL POST ke /predict
      → Hasil JSON ditampilkan di mulai_deteksi.php
```

---

## 📊 Spesifikasi Model

| Parameter | Nilai |
|-----------|-------|
| Arsitektur | CNN (Custom, 4 blok konvolusi) |
| Input | 128 × 128 piksel, 3 channel (RGB) |
| Output | 5 kelas (Softmax) |
| Batch Size | 15 |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Augmentasi | Rotasi, geser, zoom, flip, brightness |
| Early Stopping | patience=15 (monitor val_loss) |
| Format Produksi | TFLite FP16 (~17 MB) |
| Format Backup | Keras H5 (~100 MB) |

---

## ⚙️ Konfigurasi Penting

### Di `api_flask.py`
```python
MODE = os.getenv("FLASK_MODE", "online")  # "local" untuk testing
MODEL_DIR = os.path.join(BASE_DIR, "model")  # Folder model
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model_fp16.tflite")  # Path model utama
IMG_SIZE = (128, 128)  # Ukuran input — jangan diubah!
CLASS_NAMES = ["Healthy", "Bacterialblight", "Blast", "Brownspot", "Tungro"]  # Urutan kelas — jangan diubah!
```

### Di `train_colab.py`
```python
IMG_SIZE   = (128, 128)  # Harus sama dengan api_flask.py
BATCH_SIZE = 15
EPOCHS     = 100
CLASS_NAMES = ["Healthy", "Bacterialblight", "Blast", "Brownspot", "Tungro"]  # Harus sama!
```

> ⚠️ **KRITIKAL**: `CLASS_NAMES` dan `IMG_SIZE` di `api_flask.py` dan `train_colab.py`
> **HARUS PERSIS SAMA**. Jika berbeda, prediksi akan salah kelas.

---

## 📦 Dependencies (`requirements.txt`)

```
flask
tensorflow
opencv-python-headless
numpy
gunicorn
```
