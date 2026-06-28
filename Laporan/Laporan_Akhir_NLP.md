# **Pembuatan Ekstensi Browser untuk Deteksi dan Klasifikasi Komentar Judi Online pada YouTube menggunakan Natural Language Processing**

**Dosen Pengampu** : Muhammad Yazid Supriadi, S.Kom., M.Kom.
**Asisten Dosen/Lab** : Romi Wahyudi

| No | Anggota Kelompok | NIM |
|:--:|------------------|:---:|
| 1 | Muhammad Jibril Ibrahim | 0110224002 |
| 2 | Muhamad Ridwan Karim | 0110224122 |
| 3 | Anwar Maulana | 0110224020 |
| 4 | Rohmatul Hidayat | 0110224015 |
| 5 | Achmad Muflih Alrasyid | 0110224162 |

**Program Studi Teknik Informatika**
**Sekolah Tinggi Teknologi Terpadu Nurul Fikri**
**Depok**
**2026**

---

## ABSTRAK

Komentar promosi judi online (judol) pada platform YouTube semakin marak dan mengganggu pengalaman pengguna. Penelitian ini bertujuan untuk mengembangkan sistem klasifikasi komentar YouTube berbahasa Indonesia menggunakan teknik Natural Language Processing serta mengimplementasikannya dalam bentuk ekstensi browser. Tiga model klasifikasi dikembangkan dan dibandingkan: BiGRU berbasis PyTorch, LSTM berbasis TensorFlow, dan IndoBERT berbasis HuggingFace Transformers dengan Focal Loss. Dataset yang digunakan merupakan gabungan enam sumber dari Kaggle berjumlah 70.379 komentar dengan rasio ketidakseimbangan kelas 7,6:1. Hasil evaluasi menunjukkan bahwa IndoBERT dengan Focal Loss mencapai performa terbaik dengan accuracy 99,64%, precision kelas judol 0,99, recall 0,98, dan hanya 51 kesalahan total dari 14.076 sampel uji. Model tersebut kemudian diimplementasikan ke dalam ekstensi browser berbasis Manifest V3 yang mampu mendeteksi, menyembunyikan (Hide), atau menandai (Highlight) komentar judi online secara real-time pada halaman YouTube.

**Kata Kunci:** klasifikasi teks, deteksi judi online, IndoBERT, BiGRU, Focal Loss, ekstensi browser, YouTube

---

## DAFTAR ISI

1. [PENDAHULUAN](#1-pendahuluan)
   1.1 Latar Belakang
   1.2 Tujuan Penelitian
   1.3 Batasan Masalah
2. [DATASET](#2-dataset)
   2.1 Sumber Dataset
   2.2 Karakteristik Dataset
   2.3 Pembagian Data
3. [PIPELINE NATURAL LANGUAGE PROCESSING](#3-pipeline-natural-language-processing)
   3.1 Pengumpulan Dataset
   3.2 Preprocessing
   3.3 Tokenisasi
   3.4 Pelatihan Model
   3.5 Evaluasi Model
   3.6 Implementasi
4. [MODEL](#4-model)
   4.1 Gated Recurrent Unit
   4.2 Long Short-Term Memory
   4.3 IndoBERT
5. [IMPLEMENTASI SISTEM](#5-implementasi-sistem)
   5.1 Arsitektur Sistem
   5.2 Alur Implementasi Sistem
   5.3 Komponen Ekstensi Browser
   5.4 Antarmuka Ekstensi Browser
6. [HASIL DAN EVALUASI](#6-hasil-dan-evaluasi)
   6.1 Hasil Implementasi
   6.2 Hasil Pengujian Ekstensi
   6.3 Evaluasi Model
   6.4 Analisis Hasil
7. [KESIMPULAN DAN SARAN](#7-kesimpulan-dan-saran)
   7.1 Kesimpulan
   7.2 Saran

[DAFTAR PUSTAKA](#daftar-pustaka)

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang

YouTube merupakan platform media sosial yang menyediakan fitur komentar sebagai sarana interaksi antar pengguna. Namun, fitur ini sering dimanfaatkan untuk menyebarkan promosi judi online (judol) melalui komentar yang berisi ajakan bermain, tautan situs, maupun klaim keuntungan. Komentar tersebut umumnya dibuat secara berulang dengan variasi penulisan, penggunaan emoji, karakter Unicode, dan bahasa tidak baku sehingga sulit dideteksi maupun difilter secara manual. Promosi judi online di kolom komentar YouTube tidak hanya mengganggu pengalaman menonton, tetapi juga berpotensi menyesatkan pengguna, terutama remaja.

Natural Language Processing (NLP) merupakan salah satu bidang kecerdasan buatan yang mampu mengolah dan memahami teks secara otomatis. Dengan memanfaatkan teknik klasifikasi teks (text classification), komentar dapat diklasifikasikan menjadi dua kategori, yaitu judol dan bukan judol. Dalam beberapa tahun terakhir, pendekatan berbasis Recurrent Neural Network (RNN) seperti Gated Recurrent Unit (GRU) [1] dan Long Short-Term Memory (LSTM) [2] telah banyak digunakan untuk tugas klasifikasi teks. Selain itu, pendekatan berbasis Transformer seperti BERT (Bidirectional Encoder Representations from Transformers) [3] menunjukkan performa yang lebih unggul dalam memahami konteks kalimat. Untuk menangani ketidakseimbangan kelas, Focal Loss [4] telah terbukti efektif dalam memfokuskan pembelajaran pada sampel yang sulit diklasifikasikan.

Pada penelitian ini digunakan tiga pendekatan model: BiGRU dan LSTM sebagai model berbasis RNN, serta IndoBERT [5] sebagai model berbasis Transformer yang telah dilatih pada korpus bahasa Indonesia. Model dengan performa terbaik kemudian diimplementasikan ke dalam ekstensi browser yang dapat mendeteksi serta menyembunyikan atau menandai komentar yang terindikasi sebagai promosi judi online secara otomatis.

### 1.2 Tujuan Penelitian

Penelitian ini bertujuan untuk:

1. Mengembangkan sistem klasifikasi komentar YouTube berbahasa Indonesia menggunakan teknik Natural Language Processing.
2. Membandingkan performa tiga model -- BiGRU, LSTM, dan IndoBERT -- dalam mendeteksi komentar judi online.
3. Mengimplementasikan model terbaik ke dalam ekstensi browser berbasis Manifest V3 untuk mendeteksi dan menyaring komentar judi online pada YouTube secara real-time.

### 1.3 Batasan Masalah

Batasan masalah pada penelitian ini adalah:

1. Data yang digunakan terbatas pada komentar berbahasa Indonesia.
2. Deteksi hanya dilakukan pada teks komentar, bukan pada gambar, video, atau tautan.
3. Ekstensi browser hanya mendukung platform YouTube.
4. Model dijalankan secara online melalui API (tidak ada inferensi lokal di sisi klien).

---

## 2. DATASET

### 2.1 Sumber Dataset

Dataset yang digunakan dalam penelitian ini merupakan gabungan dari enam dataset publik yang tersedia di platform Kaggle. Penggunaan beberapa dataset bertujuan untuk memperoleh variasi komentar yang lebih beragam sehingga model dapat mengenali berbagai pola komentar promosi judi online. Seluruh dataset berisi komentar berbahasa Indonesia yang telah diberi label menjadi dua kelas, yaitu judol dan bukan judol. Sumber dataset ditunjukkan pada Tabel 1.

**Tabel 1. Sumber Dataset**

| No | Nama Dataset | Kontributor |
|:--:|--------------|:-----------:|
| 1 | Judi Online | Yaemico |
| 2 | Deteksi Judi Online | Yaemico |
| 3 | Dataset Komentar Judi Online di YouTube | Kyyyy8 |
| 4 | Komentar Judi Online | Fahruu |
| 5 | Dataset Komentar Judi Online Platform YouTube | Kyyyy8 |
| 6 | Dataset Scraping Komentar Judi Online | A. Kalindra |

### 2.2 Karakteristik Dataset

Setelah melalui tahap penggabungan, pembersihan, dan deduplikasi, diperoleh dataset akhir dengan karakteristik sebagai berikut (Tabel 2).

**Tabel 2. Karakteristik Dataset**

| Metrik | Nilai |
|--------|-------|
| Total baris | 70.379 |
| Label 0 (Bukan Judol) | 62.202 (88,4%) |
| Label 1 (Judol) | 8.177 (11,6%) |
| Kolom | text, label |
| Rata-rata panjang teks | ~11 kata |
| Persentil ke-95 panjang teks | 31 kata |
| Persentil ke-99 panjang teks | 66 kata |

Dataset mengalami ketidakseimbangan kelas (class imbalance) yang signifikan dengan rasio sekitar 7,6:1 antara kelas bukan judol dan judol. Oleh karena itu, diperlukan teknik mitigasi seperti class weights atau focal loss [4] pada saat pelatihan model.

Berdasarkan analisis eksploratori data (EDA), ciri khas komentar judol meliputi: nama situs tersamar (contoh: alexis17, pulauwin, sgi88), kata promosi (contoh: depo, wd, gacor, maxwin), emoji tertentu (contoh: red_heart, fire, star), serta pola kalimat yang cenderung seragam (template-based). Sebaliknya, komentar bukan judol didominasi oleh bahasa percakapan sehari-hari dengan variasi panjang teks yang lebih lebar.

### 2.3 Pembagian Data

Dataset dibagi menjadi data latih dan data uji dengan teknik stratified sampling untuk menjaga proporsi kelas. Detail pembagian untuk masing-masing model ditunjukkan pada Tabel 3.

**Tabel 3. Pembagian Data**

| Model | Rasio Split | Random State | Rincian |
|-------|-------------|:------------:|---------|
| BiGRU | 80:20 | 42 | Train: 56.303, Test: 14.076 |
| LSTM | 70:15:15 | 42 | Train: 49.265, Val: 7.038, Test: 14.076 |
| IndoBERT-focal | 80:20 | 42 | Train: 56.303, Test: 14.076 |

BiGRU dan IndoBERT-focal menggunakan split yang identik (80:20, random_state=42) sehingga perbandingan performa antar keduanya bersifat adil. LSTM menggunakan validation set terpisah dengan rasio 70:15:15, namun ukuran test set tetap sama (14.076 baris).

---

## 3. PIPELINE NATURAL LANGUAGE PROCESSING

Pipeline NLP pada proyek ini terdiri dari enam tahap utama yang dijalankan secara berurutan: pengumpulan dataset, preprocessing, tokenisasi, pelatihan model, evaluasi model, dan implementasi.

### 3.1 Pengumpulan Dataset

Dataset mentah dari enam sumber digabungkan menggunakan library pandas [9] pada Notebook DataPreparation.ipynb. Proses penggabungan meliputi: (a) loading masing-masing file CSV, (b) standardisasi kolom menjadi text (isi komentar) dan label (0/1), (c) deduplikasi berdasarkan teks komentar, (d) validasi label, dan (e) penyimpanan hasil ke dataset_prepared.csv.

### 3.2 Preprocessing

Preprocessing dilakukan melalui dua lapisan (layer) yang identik antara proses pelatihan (Python) dan runtime ekstensi (JavaScript), sehingga konsistensi data antara training dan inferensi tetap terjaga. Pipeline preprocessing diimplementasikan dengan mengacu pada strategi yang dirancang untuk menangani karakteristik unik komentar judol [5].

**Layer 1 -- Advanced Preprocessing:**

Langkah-langkah preprocessing secara berurutan adalah sebagai berikut:

1. Demojize: Mengubah emoji menjadi deskripsi teks menggunakan library emoji. Tahap ini harus dilakukan paling awal agar emoji tidak terhapus oleh proses pembersihan ASCII.
2. Unicode Normalization (NFKD): Menormalkan font fancy Unicode ke bentuk dasar ASCII (contoh: \xed\x8e\x99\xed\xb0\ x8e\xed\x99\xa5\xed\x99\ x8e menjadi "situs").
3. URL Masking: Menyamarkan tautan (http, https, bit.ly) menjadi token [URL].
4. Lowercasing: Standarisasi kapitalisasi.
5. Symbol Cleaning: Menghapus simbol non-alfanumerik.
6. Slang Normalisasi: Menormalkan singkatan dan bahasa tidak baku menggunakan library saka-nlp [10].
7. Reduksi Repetisi: Mengompres karakter berulang lebih dari 2 kali (contoh: "bngettt" menjadi "bngett").
8. Whitespace Cleanup: Menghapus spasi berlebih.

**Layer 2 -- GRU-specific Cleaning (khusus BiGRU):**

Layer tambahan untuk menyesuaikan dengan vocab model BiGRU, meliputi masking URL ulang dan normalisasi slang tambahan menggunakan kamus khusus (SLANG_JUDOL_MAP) yang dibangun dari data latih.

Pipeline preprocessing lengkap diimplementasikan dalam fungsi final_preprocessing_pipeline() pada Notebook AdvancedPreprocessing.ipynb. Di sisi ekstensi browser, pipeline yang sama diimplementasikan dalam preprocess.js menggunakan kamus emoji dan slang yang disalin dari Python.

### 3.3 Tokenisasi

Teks yang telah dipreprocessing dikonversi menjadi representasi numerik melalui tokenisasi. Pendekatan tokenisasi dibedakan berdasarkan jenis model:

**a) Word-Level Tokenization (untuk BiGRU dan LSTM)**

Kata-kata dalam teks dipetakan menjadi ID unik berdasarkan vocabulary yang dibangun dari seluruh data latih.

**Tabel 4. Parameter Tokenisasi Word-Level**

| Parameter | BiGRU | LSTM |
|-----------|:-----:|:----:|
| Vocab Size | 23.154 | 15.000 |
| Max Length | 50 | 100 |
| OOV Token | <UNK> | <OOV> |
| Padding | Post-padding | Post-padding |

**b) Subword Tokenization (untuk IndoBERT)**

Menggunakan tokenizer bawaan dari model indobenchmark/indobert-base-p1 [5] dengan algoritma WordPiece (~32K vocabulary). Parameter yang digunakan: max length 128 token, truncation enabled, padding to max length.

### 3.4 Pelatihan Model

Tiga model dilatih secara terpisah menggunakan framework yang berbeda: BiGRU [1] dengan PyTorch [6] pada GPU NVIDIA RTX 3050 6GB, LSTM [2] dengan TensorFlow/Keras [7] pada CPU, dan IndoBERT [5] dengan HuggingFace Transformers [8] pada GPU.

Detail konfigurasi masing-masing model dijelaskan pada Bagian 4.

### 3.5 Evaluasi Model

Setiap model dievaluasi menggunakan metrik accuracy, precision, recall, F1-score, confusion matrix, serta analisis false positive dan false negative [9]. LSTM tambahan melaporkan ROC-AUC Score.

### 3.6 Implementasi

Model dengan performa terbaik (IndoBERT-focal) dan alternatif ringan (BiGRU) dideploy sebagai layanan API menggunakan FastAPI pada platform Hugging Face Spaces. Ekstensi browser pada Chrome/Edge (Manifest V3) mengirimkan komentar yang telah dipreprocessing ke API untuk mendapatkan prediksi, kemudian menerapkan aksi Hide atau Highlight sesuai pengaturan pengguna.

---

## 4. MODEL

### 4.1 Gated Recurrent Unit

Gated Recurrent Unit (GRU) merupakan pengembangan dari Recurrent Neural Network yang dirancang untuk mengatasi masalah vanishing gradient [1]. GRU menggunakan dua mekanisme -- update gate dan reset gate -- untuk mengatur informasi yang dipertahankan maupun diperbarui selama proses pembelajaran. Dibandingkan LSTM, GRU memiliki jumlah parameter yang lebih sedikit sehingga proses pelatihan lebih cepat dengan performa yang tetap baik pada tugas klasifikasi teks.

Pada penelitian ini, BiGRU (Bidirectional GRU) digunakan dengan konfigurasi yang ditunjukkan pada Tabel 5. Model dilatih from scratch (tanpa pre-trained weights) selama 5 epoch.

**Tabel 5. Konfigurasi BiGRU**

| Parameter | Nilai |
|-----------|-------|
| Framework | PyTorch |
| Arsitektur | Bidirectional GRU (2 layer) |
| Embedding Dimension | 128 |
| Hidden Dimension | 64 |
| Dropout | 0,3 |
| Loss Function | BCEWithLogitsLoss |
| Optimizer | Adam (lr=0,001) |
| Batch Size | 64 |
| Epochs | 5 |
| Device | GPU (NVIDIA RTX 3050 6GB) |

### 4.2 Long Short-Term Memory

Long Short-Term Memory (LSTM) merupakan pengembangan dari RNN yang memiliki tiga mekanisme gerbang -- forget gate, input gate, dan output gate -- untuk menyimpan informasi jangka panjang [2]. LSTM banyak digunakan dalam tugas NLP karena mampu mempelajari ketergantungan jarak jauh (long-range dependencies) antar kata dalam suatu kalimat.

Pada penelitian ini, LSTM digunakan sebagai model pembanding dengan konfigurasi yang ditunjukkan pada Tabel 6.

**Tabel 6. Konfigurasi LSTM**

| Parameter | Nilai |
|-----------|-------|
| Framework | TensorFlow / Keras |
| Arsitektur | Bidirectional LSTM (2 layer: 64 unit, 32 unit) |
| Embedding Dimension | 128 |
| Max Vocab | 15.000 kata |
| Max Length | 100 token |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam (lr=0,001) |
| Class Weights | {0: 0,567, 1: 4,303} |
| Batch Size | 128 |
| Max Epochs | 20 (Early Stopping, patience=4) |
| Device | CPU |

LSTM menggunakan class weights untuk mengatasi ketidakseimbangan kelas, dengan bobot lebih besar pada kelas judol (1: 4,303).

### 4.3 IndoBERT

IndoBERT merupakan model berbasis arsitektur Transformer yang telah dilatih menggunakan korpus teks berbahasa Indonesia oleh Koto et al. [5]. Model ini memanfaatkan mekanisme self-attention sehingga mampu memahami konteks kalimat secara lebih baik dibandingkan model RNN.

Pada penelitian ini, IndoBERT digunakan melalui proses fine-tuning dengan konfigurasi yang ditunjukkan pada Tabel 7.

**Tabel 7. Konfigurasi IndoBERT-focal**

| Parameter | Nilai |
|-----------|-------|
| Framework | HuggingFace Transformers |
| Pre-trained Model | indobenchmark/indobert-base-p1 |
| Num Labels | 2 |
| Max Length | 128 token |
| Loss Function | Focal Loss (alpha=[1,0; 7,61], gamma=2,0) |
| Optimizer | AdamW (lr=2e-5, weight_decay=0,01) |
| Batch Size | 32 (train) / 64 (eval) |
| Epochs | 5 |
| FP16 | Enabled |
| Training Runtime | ~52 menit (3.110 detik) |
| Device | GPU (NVIDIA RTX 3050 6GB) |

Focal Loss [4] dipilih untuk mengatasi ketidakseimbangan kelas. Berbeda dari Cross-Entropy standar, Focal Loss memodifikasi loss dengan faktor (1 - p_t)^gamma yang mengecilkan kontribusi loss dari sampel yang mudah diklasifikasi (kelas mayoritas) dan membesarkan loss dari sampel yang sulit (kelas minoritas judol). Bobot alpha [1,0; 7,61] memberikan penalti lebih besar untuk kesalahan pada kelas minoritas, dengan rasio sesuai proporsi dataset.

---

## 5. IMPLEMENTASI SISTEM

### 5.1 Arsitektur Sistem

Setelah proses pelatihan dan evaluasi model selesai, dua model dideploy sebagai layanan API menggunakan FastAPI pada platform Hugging Face Spaces:

1. IndoBERT-focal (model utama -- performa terbaik).
2. BiGRU (model alternatif -- lebih ringan).

Arsitektur sistem secara keseluruhan ditunjukkan pada Gambar 1.

```
 +---------------+     +--------------------+     +-----------------------+
 | YouTube DOM   |---->| Content Script     |---->| Service Worker        |
 | comments      |     | (youtube.js)       |     | (background.js)       |
 +---------------+     | + preprocess.js    |     +-----------+-----------+
         ^             +--------------------+                 | POST /predict
         |                                                      v
         |                                               +---------------+
         +-----------------------------------------------| HF Spaces API |
                 Hide / Highlight action                 | (FastAPI)     |
                                                         +---------------+
```

**Gambar 1. Arsitektur Sistem**

### 5.2 Alur Implementasi Sistem

Proses implementasi sistem berlangsung secara otomatis ketika pengguna membuka halaman video YouTube. Alur implementasi sistem adalah sebagai berikut:

1. Pengguna mengaktifkan ekstensi browser.
2. Content script (youtube.js) memindai komentar menggunakan MutationObserver untuk menangani konten yang dimuat secara dinamis.
3. Setiap komentar diproses menggunakan pipeline preprocessing JavaScript (preprocess.js) yang identik dengan pipeline pelatihan model.
4. Hasil preprocessing dikirimkan secara batch ke Service Worker.
5. Service Worker mengirimkan permintaan HTTP POST ke API Hugging Face Spaces.
6. API menjalankan inferensi model dan mengembalikan label prediksi beserta nilai confidence.
7. Ekstensi menerapkan aksi sesuai mode yang dipilih:
   - Mode Hide: menyembunyikan komentar terdeteksi judol.
   - Mode Highlight: menandai komentar dengan border dan label confidence.
8. Statistik deteksi diperbarui pada popup extension.

### 5.3 Komponen Ekstensi Browser

Ekstensi browser dikembangkan menggunakan Chrome Extension Manifest V3 dengan komponen utama yang ditunjukkan pada Tabel 8.

**Tabel 8. Komponen Ekstensi Browser**

| Komponen | Berkas | Fungsi |
|----------|--------|--------|
| Content Script | content/youtube.js | Mengambil komentar dari DOM YouTube, menerapkan hide/highlight |
| Preprocessing Module | content/preprocess.js | Preprocessing teks sesuai pipeline pelatihan |
| Service Worker | background/background.js | Menghubungkan ekstensi dengan API melalui HTTP request |
| Popup UI | popup/popup.html, popup.js | Pengaturan dan statistik deteksi |
| Styles | styles/ | CSS untuk efek hide dan highlight |

Setiap komponen bekerja secara independen namun saling terintegrasi melalui message passing yang disediakan oleh Chrome Extension API.

### 5.4 Antarmuka Ekstensi Browser

Ekstensi browser menyediakan antarmuka popup yang memudahkan pengguna dalam mengatur proses deteksi komentar judi online. Fitur-fitur yang tersedia meliputi:

1. Enable/Disable Extension: Master switch untuk mengaktifkan atau menonaktifkan deteksi tanpa uninstall.
2. Mode Hide: Menyembunyikan komentar terdeteksi judol dari tampilan.
3. Mode Highlight: Menandai komentar terdeteksi dengan border berwarna dan label "TERDETEKSI JUDOL" beserta nilai confidence.
4. Confidence Threshold: Slider (0,50-0,95) untuk batas minimum keyakinan model.
5. API Status: Indikator koneksi antara ekstensi dengan layanan API.
6. Statistik Deteksi: Jumlah komentar terdeteksi pada halaman yang sedang dibuka.

---

## 6. HASIL DAN EVALUASI

### 6.1 Hasil Implementasi

Setelah ekstensi browser berhasil diimplementasikan, dilakukan pengujian pada beberapa halaman video YouTube yang mengandung komentar dengan karakteristik berbeda. Ekstensi mampu mendeteksi komentar secara otomatis ketika komentar dimuat pada halaman tanpa memerlukan tindakan manual dari pengguna.

Salah satu hasil implementasi menunjukkan bahwa ekstensi berhasil mengidentifikasi komentar yang mengandung unsur promosi judi online. Komentar tersebut mengandung kata kunci seperti bonus deposit, Zeus, dan gampang menang yang umum digunakan dalam promosi situs judi online. Model mengklasifikasikan komentar tersebut sebagai TERDETEKSI JUDOL dengan nilai confidence yang sangat tinggi (99,62%).

### 6.2 Hasil Pengujian Ekstensi

Pengujian dilakukan untuk mengetahui kemampuan ekstensi browser dalam mendeteksi komentar judi online secara otomatis. Berdasarkan hasil pengujian, ekstensi mampu:

- Mendeteksi komentar yang mengandung unsur promosi judi online secara otomatis.
- Menampilkan hasil prediksi beserta nilai confidence.
- Menjalankan aksi Hide maupun Highlight sesuai pengaturan pengguna.
- Memproses komentar yang baru dimuat (dynamic loading) tanpa refresh halaman.

Selama pengujian, ekstensi bekerja dengan baik tanpa mengganggu proses pemuatan halaman YouTube. MutationObserver memungkinkan deteksi berjalan secara real-time saat pengguna melakukan scroll pada halaman komentar.

### 6.3 Evaluasi Model

Evaluasi dilakukan terhadap ketiga model yang dilatih -- BiGRU, LSTM, dan IndoBERT-focal -- pada dataset uji yang identik (14.076 sampel). Hasil evaluasi dirangkum pada Tabel 9.

**Tabel 9. Perbandingan Metrik Antar Model**

| Metrik | BiGRU | LSTM | IndoBERT-focal |
|--------|:-----:|:----:|:--------------:|
| Accuracy | 99,23% | 99% | 99,64% |
| Judol Precision | 0,97 | 0,93 | 0,99 |
| Judol Recall | 0,96 | 0,97 | 0,98 |
| Judol F1-Score | 0,97 | 0,95 | 0,98 |
| False Positive | 49 | 119 | 23 |
| False Negative | 65 | 49 | 28 |
| ROC-AUC | -- | 0,9953 | -- |

Perbandingan arsitektur ketiga model ditunjukkan pada Tabel 10.

**Tabel 10. Perbandingan Arsitektur Model**

| Aspek | BiGRU | LSTM | IndoBERT-focal |
|-------|:-----:|:----:|:--------------:|
| Jenis Model | RNN (GRU) | RNN (LSTM) | Transformer (BERT) |
| Framework | PyTorch | TensorFlow/Keras | HuggingFace Transformers |
| Pre-trained | Tidak | Tidak | Ya (indobert-base-p1) |
| Tokenisasi | Word-level (23K) | Word-level (15K) | Subword WordPiece (32K) |
| Penanganan Imbalance | -- | Class Weights | Focal Loss |
| Epochs Aktual | 5 | 6 (early stop) | 5 |

Confusion matrix untuk masing-masing model ditunjukkan pada Tabel 11.

**Tabel 11. Confusion Matrix**

| Model | | Prediksi: Bukan Judol | Prediksi: Judol |
|-------|----------------------|:---------------------:|:---------------:|
| BiGRU | Aktual: Bukan Judol | 12.392 (TN) | 49 (FP) |
| | Aktual: Judol | 65 (FN) | 1.570 (TP) |
| LSTM | Aktual: Bukan Judol | 12.322 (TN) | 119 (FP) |
| | Aktual: Judol | 49 (FN) | 1.586 (TP) |
| IndoBERT | Aktual: Bukan Judol | 12.418 (TN) | 23 (FP) |
| | Aktual: Judol | 28 (FN) | 1.607 (TP) |

Analisis per model:

**IndoBERT-focal** mencapai performa tertinggi dengan accuracy 99,64% -- hanya 23 False Positive dan 28 False Negative dari 14.076 sampel uji. Focal Loss dengan alpha=7,61 dan gamma=2,0 terbukti efektif mengatasi ketidakseimbangan kelas. Analisis error menunjukkan bahwa sebagian besar kesalahan model kemungkinan disebabkan oleh noise label dataset, bukan kelemahan model. Sebagai contoh, seluruh 28 sampel False Negative memiliki confidence tinggi (>0,98) dan secara visual tidak mengandung unsur judol, mengindikasikan bahwa sampel tersebut adalah labeling error di dataset (komentar normal yang salah dilabeli sebagai judol).

**BiGRU** mencapai accuracy 99,23% dengan 49 False Positive dan 65 False Negative. Model ini jauh lebih ringan dari IndoBERT (parameter lebih sedikit), menjadikannya alternatif yang cocok untuk lingkungan dengan sumber daya terbatas.

**LSTM** memiliki recall Judol tertinggi (0,97) setelah IndoBERT, dengan ROC-AUC 0,9953 yang menunjukkan kemampuan diskriminasi kelas yang sangat baik. Namun, precision Judol lebih rendah (0,93) dengan 119 False Positive.

### 6.4 Analisis Hasil

Hasil implementasi menunjukkan bahwa sistem mampu menjalankan fungsi utama sebagai penyaring komentar judi online pada platform YouTube.

Faktor-faktor yang mendukung keberhasilan deteksi:

1. Dataset yang beragam dari enam sumber memungkinkan model mempelajari variasi komentar yang lebih luas.
2. Pipeline preprocessing yang komprehensif mampu menormalkan emoji, Unicode, slang, URL, dan karakter berulang.
3. Konsistensi pipeline preprocessing antara Python (training) dan JavaScript (runtime) menjaga konsistensi hasil prediksi.
4. Focal Loss pada IndoBERT berhasil mengatasi ketidakseimbangan kelas [4].

Keterbatasan sistem meliputi: (a) ketergantungan pada koneksi internet, (b) akurasi dapat menurun pada komentar dengan istilah baru atau penyamaran kata yang belum ada di kamus normalisasi, dan (c) ketergantungan pada struktur DOM YouTube yang dapat berubah sewaktu-waktu.

---

## 7. KESIMPULAN DAN SARAN

### 7.1 Kesimpulan

Berdasarkan hasil penelitian dan implementasi yang telah dilakukan, dapat disimpulkan beberapa hal sebagai berikut:

1. Sistem pendeteksi komentar judi online berbasis Natural Language Processing berhasil dikembangkan dalam bentuk ekstensi browser (Chrome/Edge, Manifest V3) yang dapat mendeteksi komentar promosi judi online pada YouTube secara otomatis. Sistem mencakup tahapan preprocessing, klasifikasi menggunakan model NLP, serta penerapan aksi Hide atau Highlight sesuai pengaturan pengguna.

2. Perbandingan tiga model klasifikasi menunjukkan bahwa IndoBERT-focal memberikan performa terbaik secara keseluruhan dengan accuracy 99,64%, Judol Precision 0,99, Judol Recall 0,98, dan hanya 23 False Positive serta 28 False Negative dari 14.076 sampel uji. Model BiGRU menyusul dengan accuracy 99,23% (49 FP, 65 FN), dan LSTM dengan accuracy 99% (119 FP, 49 FN, ROC-AUC 0,9953).

3. Focal Loss terbukti efektif dalam mengatasi ketidakseimbangan kelas (rasio ~7,6:1). Dengan alpha=7,61 dan gamma=2,0, IndoBERT-focal mampu memfokuskan pembelajaran pada sampel kelas minoritas tanpa mengorbankan spesifisitas terhadap kelas mayoritas.

4. Implementasi ekstensi browser berhasil membuktikan bahwa model NLP yang telah dilatih dapat diterapkan pada lingkungan penggunaan nyata. Pengujian menunjukkan bahwa ekstensi mampu mendeteksi komentar judi online dengan tingkat keyakinan tinggi secara real-time tanpa mengganggu pengalaman pengguna.

5. Konsistensi pipeline preprocessing antara Python (training) dan JavaScript (runtime) menjadi faktor kunci dalam menjaga akurasi prediksi pada saat inferensi.

### 7.2 Saran

Beberapa aspek yang dapat dikembangkan pada penelitian selanjutnya:

1. Dataset: Menambahkan dataset yang lebih beragam dan mutakhir agar model mampu mengenali pola komentar judi online terbaru. Review label pada 51 sampel error (23 FP + 28 FN) direkomendasikan untuk memvalidasi noise label.

2. Ketahanan model: Mengembangkan model agar lebih tahan terhadap teknik penyamaran kata, penggunaan simbol, karakter Unicode, maupun kombinasi emoji yang sering digunakan untuk menghindari deteksi.

3. Optimasi performa: Mengoptimalkan kecepatan inferensi ekstensi browser, terutama saat memproses komentar dalam jumlah besar. Opsi yang dapat dipertimbangkan meliputi penggunaan ONNX Runtime di browser atau inferensi offline dengan model ringan (BiGRU).

4. Fitur pelaporan: Menambahkan fitur pelaporan (report) komentar yang terdeteksi ke platform YouTube.

5. Deteksi multimodal: Mengembangkan sistem agar tidak hanya mendeteksi teks, tetapi juga menganalisis gambar profil atau tautan yang berpotensi digunakan sebagai media promosi.

6. Pembaruan berkala: Melakukan pelatihan ulang model secara berkala dengan data baru agar sistem tetap adaptif terhadap perubahan pola promosi judi online.

---

## DAFTAR PUSTAKA

[1] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," arXiv preprint arXiv:1406.1078, 2014.

[2] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[3] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proceedings of NAACL-HLT, 2019, pp. 4171-4186.

[4] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal Loss for Dense Object Detection," in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980-2988.

[5] F. Koto, A. Rahimi, J. H. Lau, and T. Baldwin, "IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP," in Proceedings of the 28th International Conference on Computational Linguistics (COLING), 2020, pp. 557-570.

[6] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in Advances in Neural Information Processing Systems 32 (NeurIPS), 2019, pp. 8024-8035.

[7] M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning," in Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2016, pp. 265-283.

[8] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020, pp. 38-45.

[9] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[10] Saka NLP, "saka: Indonesian Text Normalization Library," Python Package Index (PyPI), 2024. [Online]. Available: https://pypi.org/project/saka/.