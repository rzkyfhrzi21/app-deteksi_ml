#!/bin/bash
# ============================================================
# FILE: start.sh
# TUJUAN: Skrip perintah untuk menghidupkan server Flask di Render.com
#
# Ibarat "tombol ON untuk mesin server" — file ini dijalankan
# secara otomatis oleh Render.com setiap kali server dinyalakan.
# Render membaca file ini dan menjalankan perintahnya untuk
# menghidupkan API Flask kita menggunakan Gunicorn (server produksi).
#
# KENAPA GUNICORN, BUKAN 'python api_flask.py' BIASA?
# Karena Flask bawaan (mode debug) tidak cocok untuk produksi —
# hanya bisa melayani 1 permintaan sekaligus dan tidak stabil.
# Gunicorn adalah "manajer" yang lebih handal untuk server sungguhan.
# ============================================================

# (1) Beritahu Flask bahwa ini mode ONLINE (bukan local testing)
#     Variabel ini dibaca oleh api_flask.py untuk menentukan konfigurasi
export FLASK_MODE=online

# (2) Jalankan server Flask menggunakan Gunicorn dengan pengaturan berikut:
gunicorn api_flask:app \      # Jalankan objek 'app' yang ada di file 'api_flask.py'
  --workers 1 \               # Gunakan 1 proses pekerja (sesuai batasan RAM Render.com free tier)
  --threads 1 \               # Gunakan 1 thread per pekerja (cukup untuk beban ringan)
  --bind 0.0.0.0:$PORT \      # Dengarkan di semua alamat IP pada port yang diberikan Render ($PORT = otomatis)
  --timeout 120               # Beri waktu 120 detik untuk memproses 1 permintaan (karena model AI butuh waktu)