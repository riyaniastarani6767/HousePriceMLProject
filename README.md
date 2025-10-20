🏪 Superstore Analytics & Machine Learning Dashboard

Aplikasi dashboard interaktif untuk analisis penjualan dan profit pada dataset Superstore, sekaligus prediksi profitabilitas menggunakan model Machine Learning.
Dibangun menggunakan Streamlit, Pandas, dan Scikit-Learn.

Dataset link: https://www.kaggle.com/datasets/juhi1994/superstore/data

Deskripsi Singkat

Proyek ini bertujuan untuk:
1. Melakukan analisis eksploratif (EDA) terhadap data Superstore.
2. Memberikan visualisasi bisnis (penjualan, profit, kategori, tren bulanan, dll).
3. Menyediakan fitur prediksi profit berdasarkan atribut transaksi.
4. Memungkinkan prediksi batch CSV sekaligus untuk banyak data.
5. Aplikasi ini dapat digunakan baik untuk analisis interaktif maupun sebagai alat bantu pengambilan keputusan bisnis.

Instalasi dan Menjalankan Proyek
1️⃣ Clone atau download repository ini
git clone <URL_REPO_KAMU>
cd "Project Sales ML"

2️⃣ Buat dan aktifkan virtual environment
python -m venv .venv
source .venv/bin/activate       # untuk macOS / Linux
.venv\Scripts\activate          # untuk Windows

3️⃣ Install dependensi
pip install -r requirements.txt

4️⃣ Jalankan training model (sekali saja)
export PYTHONPATH="$(pwd)"      # agar src bisa diimport
python -m notebooks.train_superstore

Model hasil training akan tersimpan di:
models/model_profit_clf.joblib

5️⃣ Jalankan aplikasi Streamlit
streamlit run app/app.py

Lalu buka browser di:
http://localhost:8501

🧭 Panduan Penggunaan Aplikasi
Halaman Utama (Dashboard)
1. Upload file CSV (; separator, , decimal).
2. Gunakan Filter Global (Rentang Tanggal & Region).
3. Lihat Kartu Statistik: total baris, order unik, periode, sumber data.
4. Navigasi ke halaman:5.EDA Analysis
5. Profit Prediction
6. Lihat pratinjau data aktif & unduh hasil terfilter.

📊 Halaman EDA Analysis

Fitur:
1. KPI Cards: Total Sales, Profit, Orders, Profit Ratio.
2. Filter per Category dan Sub-Category.
Grafik:
- Sales & Profit per Category
- Top 10 Sub-Category
- Tren Bulanan Sales & Profit
- Scatter Sales vs Profit
- Tabel ringkasan 10 Sub-Category teratas + tombol download CSV.

Halaman Profit Prediction

Terdiri dari dua mode:

1. Prediksi Tunggal
Input manual fitur:
Sales, Quantity, Discount, Ship Mode, Segment, Category, Sub-Category, Region, Days_to_Ship.
Klik Prediksi → hasil berupa:
Label: Profitable / Not Profitable
Probabilitas keuntungan.

2. Prediksi Batch (CSV)
Klik Unduh Template CSV.
Isi data sesuai contoh (pemisah ;, desimal ,).
Upload kembali CSV tersebut.
Sistem otomatis:
Menghitung Days_to_Ship jika tidak ada (berdasarkan Order & Ship Date).
Menampilkan hasil prediksi + probabilitas.
Menyediakan tombol Unduh Hasil (CSV).

Fitur Utama

| Fitur                | Keterangan                               |
| -------------------- | ---------------------------------------- |
| **Data Upload**      | CSV upload dengan validasi format        |
| **EDA Visuals**      | Plotly charts interaktif                 |
| **KPI Cards**        | Ringkasan bisnis dengan warna tematik    |
| **Machine Learning** | Model klasifikasi profit (Random Forest) |
| **Batch Prediction** | Prediksi massal dari file CSV            |
| **Download CSV**     | Ekspor hasil analisis & prediksi         |

🧠 Model Machine Learning

Algoritma: RandomForestClassifier
Fitur Input:
Sales, Quantity, Discount, Ship Mode, Segment, Category, Sub-Category, Region, Days_to_Ship
Target: Profitability (1 = Profitable, 0 = Not Profitable)
File Model: models/model_profit_clf.joblib
