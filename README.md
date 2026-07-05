# 🤖 API Deteksi Penyakit Padi — Model CNN + Flask

Backend API berbasis **Python / Flask** yang menerima gambar daun padi dari aplikasi web, lalu menganalisisnya menggunakan **model kecerdasan buatan (CNN — Convolutional Neural Network)** untuk menentukan jenis penyakitnya.

Repo ini adalah "otak" dari sistem deteksi. Repo aplikasi web PHP ada di: `app-deteksi`.

---

## 🗂️ Struktur & Penjelasan Setiap File

```
app-deteksi_ml/
├── api_flask.py          → Server API utama. Menerima gambar, prediksi, kembalikan JSON
├── train_colab.py        → Script training model CNN (dijalankan di Google Colab)
├── convert_to_tflite.py  → Script konversi model H5 → TFLite (lebih ringan untuk server)
├── start.sh              → Perintah menghidupkan server di Render.com (via Gunicorn)
├── requirements.txt      → Daftar library Python yang dibutuhkan
├── setup.py              → Konfigurasi paket/instalasi
├── model.h5              → Model Keras asli hasil training (~118 MB) — untuk backup/fallback
├── model_fp16.tflite     → Model TFLite FP16 (~29 MB) — DIPAKAI di server production
└── dataset/              → Folder dataset gambar penyakit padi
```

---

## 📄 Penjelasan Detail Setiap File

### 🔷 `api_flask.py` — Server API Utama
File paling penting di repo ini. Inilah server yang "mendengar" permintaan dari aplikasi PHP.

**Cara kerja:**
1. Server dinyalakan → model `model_fp16.tflite` dimuat ke memori
2. Aplikasi PHP mengirim gambar daun padi via HTTP POST ke endpoint `/predict`
3. Server menerima gambar → preprocessing (resize 128x128, normalize) → prediksi model
4. Server mengembalikan hasil prediksi dalam format JSON

**Endpoint yang tersedia:**

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/health` | Cek apakah server hidup. Kembalikan status & info model yang dipakai |
| `POST` | `/predict` | Kirim gambar (field: `image`), terima hasil prediksi JSON |

**Contoh response `/predict`:**
```json
{
  "label": "Bacterialblight",
  "confidence": 0.9345,
  "probs": {
    "Healthy": 0.0123,
    "Bacterialblight": 0.9345,
    "Blast": 0.0210,
    "Brownspot": 0.0187,
    "Tungro": 0.0135
  }
}
```

---

### 🔷 `train_colab.py` — Script Training Model CNN
Dijalankan di **Google Colab** (bukan di komputer lokal) karena butuh GPU dan RAM besar.

**Berisi 6 bagian (Cell/Sel):**

| Sel | Isi |
|---|---|
| SEL 1 | Download dataset dari Kaggle (2 sumber berbeda) dan verifikasi |
| SEL 2 | Split dataset: 70% training / 10% validasi / 20% testing |
| SEL 3 | Definisi arsitektur model CNN, konfigurasi callbacks, mulai training |
| SEL 4 | Evaluasi model, plot grafik akurasi & loss, confusion matrix |
| SEL 5 | Konversi model terbaik → TFLite FP16 untuk server |
| SEL 6 | Verifikasi prediksi dengan gambar sampel dari dataset |

**Menghasilkan file:**
- `best_model.h5` → Model terbaik (berdasarkan val_accuracy tertinggi selama training)
- `model_final.h5` → Model di epoch terakhir (backup saja)
- `model_fp16.tflite` → Model yang siap dipakai di server

---

### 🔷 `convert_to_tflite.py` — Konversi Model H5 → TFLite
Digunakan setelah training selesai untuk mengecilkan model agar lebih ringan di server.

**Dua opsi konversi:**

| Opsi | Ukuran | Akurasi | Rekomendasi |
|---|---|---|---|
| FP16 (Float 16-bit) | ~29 MB | ≈ sama dengan H5 original | ✅ **Ini yang dipakai** |
| INT8 (Integer 8-bit) | ~15 MB | Sedikit lebih rendah | Alternatif jika RAM server kurang |

---

### 🔷 `start.sh` — Skrip Menghidupkan Server
Dijalankan otomatis oleh Render.com saat server dinyalakan.

```bash
export FLASK_MODE=online        # Beritahu Flask bahwa ini mode produksi
gunicorn api_flask:app          # Jalankan via Gunicorn (server produksi, lebih stabil dari Flask dev server)
  --workers 1                   # 1 proses pekerja (sesuai batas Render.com free)
  --threads 1                   # 1 thread per pekerja
  --bind 0.0.0.0:$PORT          # Dengarkan di port yang diberikan Render
  --timeout 120                 # Beri waktu 120 detik per request (model AI butuh waktu)
```

---

## 🧠 Penjelasan File Model

| File | Ukuran | Sumber | Dipakai untuk |
|---|---|---|---|
| `best_model.h5` | ~118 MB | Hasil `train_colab.py` (epoch dengan val_accuracy tertinggi) | Sumber konversi ke TFLite. Backup jika TFLite tidak tersedia |
| `model_final.h5` | ~118 MB | Hasil `train_colab.py` (epoch terakhir) | Backup saja. Tidak selalu lebih baik dari best_model |
| `model_fp16.tflite` | ~29 MB | Hasil konversi `best_model.h5` via `convert_to_tflite.py` | **⭐ DIPAKAI di server production** |

> **Kenapa pakai TFLite bukan H5 langsung?**
> Model H5 (Keras) berukuran ~118 MB dan butuh RAM lebih besar. Server Render.com free hanya punya RAM 512 MB.
> TFLite FP16 hanya ~29 MB dan lebih efisien, akurasi hampir sama persis dengan versi H5-nya.

---

## 🏗️ Arsitektur Model CNN

Model dibangun menggunakan **TensorFlow / Keras** dengan arsitektur sebagai berikut:

```
Input: Gambar RGB 128 × 128 piksel
│
├── Block 1: Conv2D(32) → BatchNorm → MaxPooling(2×2)
├── Block 2: Conv2D(64) → BatchNorm → MaxPooling(2×2) → Dropout(0.3)
├── Block 3: Conv2D(128) → BatchNorm → MaxPooling(2×2) → Dropout(0.4)
├── Block 4: Conv2D(256) → BatchNorm → MaxPooling(2×2) → Dropout(0.4)
│
├── Flatten
├── Dense(512, ReLU) → BatchNorm → Dropout(0.5)
└── Dense(5, Softmax)  ← Output: 5 kelas penyakit
```

**5 Kelas Output:**
| Index | Nama di Model | Keterangan |
|---|---|---|
| 0 | `Healthy` | Daun sehat, tidak ada penyakit |
| 1 | `Bacterialblight` | Hawar bakteri |
| 2 | `Blast` | Penyakit blas / busuk leher |
| 3 | `Brownspot` | Bercak cokelat |
| 4 | `Tungro` | Virus tungro |

---

## 📊 Konfigurasi Training

| Parameter | Nilai | Keterangan |
|---|---|---|
| Ukuran gambar | 128 × 128 piksel | Input ke model CNN |
| Batch size | 15 | Jumlah gambar per iterasi |
| Max epoch | 100 | Batas maksimal, biasanya berhenti lebih awal |
| EarlyStopping | patience=15 | Berhenti jika 15 epoch berturut-turut tidak membaik |
| ReduceLROnPlateau | patience=5, factor=0.5 | Turunkan learning rate jika stagnan 5 epoch |
| Split data | 70% / 10% / 20% | Training / Validasi / Testing |

---

## 🛠️ Cara Menjalankan Lokal (Testing)

### Prasyarat
- Python 3.10+
- Library sesuai `requirements.txt`

### Langkah
```bash
# 1. Install semua library yang dibutuhkan
pip install -r requirements.txt

# 2. Jalankan server Flask dalam mode lokal (Windows)
set FLASK_MODE=local
python api_flask.py

# Server berjalan di: http://127.0.0.1:5000
```

### Test Endpoint
```bash
# Cek server hidup
curl http://127.0.0.1:5000/health

# Test prediksi gambar
curl -X POST -F "image=@/path/ke/gambar.jpg" http://127.0.0.1:5000/predict
```

---

## 🌐 Alur Training → Deploy (Ringkasan)

```
1. Jalankan train_colab.py di Google Colab
         ↓
2. Download file best_model.h5 dari Colab
         ↓
3. Jalankan convert_to_tflite.py → hasilkan model_fp16.tflite
         ↓
4. Upload model_fp16.tflite ke GitHub repo app-deteksi_ml
         ↓
5. Render.com otomatis rebuild dan deploy ulang
         ↓
6. API siap menerima gambar dari aplikasi PHP (app-deteksi)
```

---

## 📝 Catatan

- Dibuat untuk keperluan **Tugas Akhir / Skripsi**
- Training dilakukan di Google Colab (GPU T4)
- Server production: Render.com (free tier)
- Timeout server diset 120 detik untuk mengakomodasi waktu inferensi model
