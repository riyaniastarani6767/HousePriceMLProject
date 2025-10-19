🏪 Superstore Analytics & ML Dashboard
🎯 Deskripsi Project

Project ini bertujuan untuk melakukan analisis data penjualan (EDA) dan membangun model machine learning sederhana untuk memprediksi apakah suatu transaksi menghasilkan profit atau tidak.
Dashboard dibuat menggunakan Streamlit agar hasil analisis dan model dapat diakses secara interaktif.

📊 Tujuan Utama
1. Business Dashboard (EDA)
Menampilkan KPI bisnis: Total Sales, Total Profit, Jumlah Order, dan Profit Ratio (%)
Filter interaktif: Region, Category, Sub-Category, dan Rentang Tanggal

Visualisasi utama:
Sales & Profit per Category
Top-N Sub-Category berdasarkan Sales
Tren Bulanan Sales & Profit
Scatter Sales vs Profit (untuk insight diskon tinggi tapi profit rendah)

2. Machine Learning (Klasifikasi)

Tujuan: Prediksi apakah order profitable (Profit > 0) atau tidak profitable (Profit ≤ 0)
Model: RandomForestClassifier
Pipeline: Menggunakan ColumnTransformer untuk encoding fitur kategori & scaling fitur numerik
Evaluasi: Accuracy, Precision, Recall, dan F1-score
Model Output: models/model_profit_clf.joblib


🧮 Fitur dan Target Model
Jenis	Nama Kolom	Keterangan
🎯 Target	Profitable	1 jika Profit > 0, else 0
🔢 Fitur Numerik	Sales, Quantity, Discount	Data angka transaksi
🏷️ Fitur Kategorikal	Ship Mode, Segment, Category, Sub-Category, Region	Data non-numerik
📆 Fitur Waktu	Order Date, Ship Date	Akan digunakan untuk membuat fitur baru: Days_to_Ship


⚙️ Langkah Proses

Data Preparation

Baca dataset dari data/raw/USSuperstoreData.csv

Ubah format angka (koma → titik)

Buat kolom target Profitable

Tambah fitur waktu: Days_to_Ship

Train-Test Split

80% data → Training

20% data → Testing

Model Training

Gunakan RandomForestClassifier dengan pipeline preprocessing

Simpan model ke models/model_profit_clf.joblib

Evaluasi

Cek metrik: accuracy, precision, recall, F1

Analisis feature importance

Dashboard Streamlit

Halaman 1: EDA Analysis

Halaman 2: Profit Prediction

Cara Menjalankan Project
1️⃣ Instalasi Library

Pastikan sudah punya Python ≥ 3.10
Lalu jalankan:

pip install -r requirements.txt

2️⃣ Jalankan Notebook (Training)
jupyter notebook notebooks/train_superstore.ipynb

3️⃣ Jalankan Dashboard Streamlit
streamlit run app/app.py


Dashboard akan muncul di browser lokal (biasanya di http://localhost:8501
)

🧠 Library yang Digunakan

pandas – manipulasi data

numpy – operasi numerik

scikit-learn – machine learning & evaluasi

joblib – simpan/load model

plotly / matplotlib – visualisasi data

streamlit – dashboard interaktif

📌 Catatan

File dataset asli disimpan di data/raw/

Model yang sudah dilatih disimpan di models/

Folder processed/ bisa digunakan untuk menyimpan data bersih (optional)

Pastikan tidak menghapus file model_profit_clf.joblib, karena digunakan oleh dashboard prediksi.





Outline Notebook (tanpa kode)
0) Header & Tujuan

Judul cell: Superstore Analytics & Profit Classification — Notebook Training
Isi:

Tujuan: (1) EDA ringkas, (2) latih model klasifikasi “Profitable?”, (3) simpan pipeline.

Sumber data: data/raw/USSuperstoreData.csv

Output: models/model_profit_clf.joblib

1) Setup & Konfigurasi

Isi:

Versi Python & paket utama (pandas, numpy, scikit-learn, joblib, dll.)

Set random seed (catat angkanya).

Catat path file yang dipakai.

Output yang diharapkan: tabel kecil/teks versi paket.

2) Load Data (cek format)

Isi:

Jelaskan bahwa file pakai delimiter ; dan desimal koma.

Baca CSV, tampilkan 5 baris pertama.

Tampilkan df.info() dan df.head() (naratif).

Checklist:

Kolom seperti Sales, Profit terbaca sebagai string (karena koma). Itu normal—akan dibersihkan nanti.

Pastikan jumlah baris ≈ dataset asli (ribuan baris).

3) Data Dictionary Ringkas

Isi:

Daftar kolom penting dan artinya (Order ID, Order Date, Ship Date, Ship Mode, Segment, Category, Sub-Category, Region, Sales, Quantity, Discount, Profit).

Tegaskan: unit data = order-line (satu produk dalam satu order).

4) Data Cleaning

Isi langkah:

Konversi kolom numerik ber-desimal koma → titik, lalu ke float: Sales, Profit.

Pastikan Quantity integer, Discount float 0–1.

Parse tanggal untuk Order Date & Ship Date.

Buat Days_to_Ship = ShipDate − OrderDate (dalam hari).

Rapikan nama kolom (opsional: ganti spasi jadi underscore).

Buang nilai aneh: Days_to_Ship < 0 atau Discount di luar [0, 1].

Hapus duplikat baris bila ada (berdasar seluruh kolom atau kombinasi aman).

Output yang diharapkan:

df.describe() untuk numerik, df.isna().sum() untuk missing.

Ringkasan berapa baris dibuang (jika ada).

5) Buat Target Profitable

Isi:

Definisi target: Profitable = 1 jika Profit > 0, else 0.

Tampilkan proporsi kelas (berapa % profit vs rugi).

Output yang diharapkan:

Bar chart kecil/teks persentase kelas (imbalance atau tidak).

6) EDA Ringkas (untuk sanity check)

Isi komponen:

KPI: Total Sales, Total Profit, Jumlah Orders unik (berdasar Order ID), Profit Ratio (% baris dengan Profitable=1).

Grafik 1: Sales & Profit per Category (bar).

Grafik 2: Top-N Sub-Category by Sales (bar).

Grafik 3: Tren bulanan Sales & Profit (resample ke bulan dari Order Date).

Grafik 4: Scatter Sales vs Profit (lihat pola diskon tinggi → profit rendah).

Catatan EDA: 3–5 poin insight singkat (mis. kategori paling besar sales, sub-kategori top, efek diskon kasar).

7) Pilih Fitur & Siapkan X, y

Fitur yang dipakai model (jelaskan pilihan):

Numerik: Sales, Quantity, Discount, Days_to_Ship

Kategorikal: Ship Mode, Segment, Category, Sub-Category, Region

Tidak menyertakan Profit (target only).

Isi:

Definisikan daftar fitur & target secara eksplisit.

Catat jika ada kolom yang banyak kategori (Sub-Category) → akan di-One-Hot.

8) Train–Test Split

Isi:

Pembagian data 80% train, 20% test, random_state konsisten.

Jelaskan kenapa split acak cukup untuk versi pertama (nanti bisa time-based).

Output yang diharapkan:

Ukuran X_train, X_test, y_train, y_test.

9) Pipeline Preprocessing + Model

Konsep yang ditulis:

ColumnTransformer:

One-Hot Encoding untuk fitur kategorikal

“Passthrough” fitur numerik (tanpa scaling untuk RF)

Estimator: RandomForestClassifier (sebut parameter dasar yang relevan: n_estimators, max_depth jika perlu).

Tujuan: menghindari data leakage & memudahkan simpan pipeline utuh.

10) Training Model

Isi:

Fit pipeline pada data train.

Catat waktu training (opsional).

Output yang diharapkan: model terlatih di memori.

11) Evaluasi di Test Set

Isi metrik & artefak evaluasi:

Accuracy, Precision, Recall, F1 (macro atau weighted — tulis pilihan).

Confusion Matrix (interpretasi singkat: FP/ FN artinya apa di bisnis).

(Opsional) ROC-AUC.

Output yang diharapkan: tabel metrik + plot confusion matrix.

Catat 3 poin interpretasi:

Apakah model cenderung “over-predict” profit?

Mana yang lebih krusial: Precision atau Recall (dalam konteks bisnis)?

Perlu threshold tuning nanti di app atau tidak.

12) Feature Importance (ringkas)

Isi:

Tampilkan feature importance dari RandomForest (untuk gambaran awal).

Jelaskan bahwa permutation importance lebih tepercaya (opsional untuk versi berikutnya).

Output: bar chart importance 10 fitur teratas (setelah OHE bisa diringkas jika ingin).

13) Validasi Cepat terhadap Data Baru (sanity)

Isi:

Ambil 3–5 sampel baris test → tampilkan prediksi & probabilitas.

Cek apakah prediksi masuk akal berdasar Discount/Sales.

(Masih tanpa kode detail; cukup rencana narasi dan checklist hasil yang diharapkan.)

14) Simpan Pipeline Model

Isi:

Simpan pipeline (preprocess + RF) ke models/model_profit_clf.joblib.

Catat lokasi file dan ukurannya.

Output yang diharapkan: file .joblib muncul di folder models/.

15) Catatan & Next Steps

Isi:

Ringkas hasil: metrik utama & insight penting.

Daftar peningkatan berikutnya:

Time-based split,

Tambah fitur Unit_Price = Sales/Quantity & binning Discount,

Baseline LogisticRegression untuk pembanding,

Permutation importance / SHAP.

Acceptance Criteria Notebook

 df bersih: tipe numerik & tanggal sudah benar.

 Kolom Profitable terbentuk dan proporsi kelas diketahui.

 EDA menghasilkan KPI & 3–4 grafik valid.

 Train–test split sukses (ukuran data sesuai).

 Pipeline RF terlatih, metrik muncul (Accuracy/Precision/Recall/F1).

 Confusion matrix tampil & diinterpretasi.

 models/model_profit_clf.joblib tersimpan.

kalau outline ini sudah oke, next aku bikinin outline untuk file Streamlit (struktur setiap halaman & interaksi form prediksi) — juga tanpa kode. mau lanjut ke itu?