
# **Pembuatan Ekstensi Browser Deteksi dan Klasifikasi Komentar Judi Online pada YouTube**

**Dosen Pengampu :** Muhammad Yazid Supriadi, S.Kom., M.Kom.
**Asisten Dosen :** Romi Wahyudi

**Anggota Kelompok**
| Nama | NIM |
|------|:---:|
| Muhammad Jibril Ibrahim | 0110224002 |
| Muhamad Ridwan Karim | 0110224122 |
| Anwar Maulana | 0110224020 |
| Rohmatul Hidayat | 0110224015 |
| Achmad Muflih Alrasyid | 0110224162 |

**PROGRAM STUDI TEKNIK INFORMATIKA**
**SEKOLAH TINGGI TEKNOLOGI TERPADU NURUL FIKRI**
**DEPOK**
**2026**

---

## Daftar Isi

[BAB I PENDAHULUAN](#bab-i-pendahuluan)
- [1.1 Latar Belakang](#11-latar-belakang)
- [1.2 Tujuan Penelitian](#12-tujuan-penelitian)

[BAB II DATASET](#bab-ii-dataset)
- [2.1 Sumber Dataset](#21-sumber-dataset)
- [2.2 Karakteristik Dataset](#22-karakteristik-dataset)
- [2.3 Pembagian Data](#23-pembagian-data)

[BAB III PIPELINE NLP](#bab-iii-pipeline-nlp)
- [3.1 Pengumpulan Dataset](#31-pengumpulan-dataset)
- [3.2 Preprocessing](#32-preprocessing)
- [3.3 Tokenisasi](#33-tokenisasi)
- [3.4 Pelatihan Model](#34-pelatihan-model)
- [3.5 Evaluasi Model](#35-evaluasi-model)
- [3.6 Implementasi](#36-implementasi)

[BAB IV MODEL YANG DIGUNAKAN](#bab-iv-model-yang-digunakan)
- [4.1 Gated Recurrent Unit (GRU)](#41-gated-recurrent-unit-gru)
- [4.2 Long Short-Term Memory (LSTM)](#42-long-short-term-memory-lstm)
- [4.3 IndoBERT](#43-indobert)

[BAB V IMPLEMENTASI SISTEM](#bab-v-implementasi-sistem)
- [5.1 Arsitektur Sistem](#51-arsitektur-sistem)
- [5.2 Alur Implementasi Sistem](#52-alur-implementasi-sistem)
- [5.3 Implementasi Browser Extension](#53-implementasi-browser-extension)
- [5.4 Implementasi Antarmuka Browser Extension](#54-implementasi-antarmuka-browser-extension)

[BAB VI HASIL & EVALUASI](#bab-vi-hasil--evaluasi)
- [6.1 Hasil Implementasi Sistem](#61-hasil-implementasi-sistem)
- [6.2 Hasil Pengujian](#62-hasil-pengujian)
- [6.3 Evaluasi Model](#63-evaluasi-model)
- [6.4 Analisis Hasil](#64-analisis-hasil)

[BAB VII KESIMPULAN](#bab-vii-kesimpulan)
- [7.1 Kesimpulan](#71-kesimpulan)
- [7.2 Saran](#72-saran)

[DAFTAR PUSTAKA](#daftar-pustaka)

---

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

YouTube merupakan salah satu platform media sosial yang menyediakan fitur komentar sebagai sarana interaksi antar pengguna. Namun, fitur ini sering dimanfaatkan untuk menyebarkan promosi judi online (judol) melalui komentar yang berisi ajakan bermain, tautan situs, maupun klaim keuntungan. Komentar tersebut umumnya dibuat secara berulang dengan variasi penulisan, penggunaan emoji, karakter Unicode, dan bahasa tidak baku sehingga sulit dideteksi maupun difilter secara manual.

Promosi judi online di kolom komentar YouTube tidak hanya mengganggu pengalaman menonton, tetapi juga berpotensi menyesatkan pengguna, terutama remaja yang mungkin tergiur dengan iming-iming keuntungan cepat. Pola teks judol sangat bervariasi — mulai dari nama situs tersamar (*alexis17*, *pulauwin*, *sgi88*), kata promosi (*depo*, *wd*, *gacor*, *maxwin*, *cuan*), hingga penggunaan emoji tertentu (*❤️*, *🔥*, *⭐*). Variasi ini membuat pendeteksian secara manual menjadi tidak praktis.

Natural Language Processing (NLP) merupakan salah satu bidang kecerdasan buatan yang mampu mengolah dan memahami teks secara otomatis. Dengan memanfaatkan teknik klasifikasi teks (*text classification*), komentar dapat diklasifikasikan menjadi dua kategori, yaitu *judol* dan *bukan_judol*. Pendekatan ini diharapkan dapat membantu mengurangi penyebaran komentar promosi judi online pada platform YouTube.

Pada penelitian ini digunakan tiga pendekatan model: **BiGRU** dan **LSTM** sebagai model berbasis Recurrent Neural Network (RNN), serta **IndoBERT** sebagai model berbasis Transformer. Model dengan performa terbaik kemudian diimplementasikan ke dalam *browser extension* yang dapat mendeteksi serta menyembunyikan atau menandai komentar yang terindikasi sebagai promosi judi online secara otomatis.

## 1.2 Tujuan Penelitian

Penelitian ini bertujuan untuk:

1. Mengembangkan sistem klasifikasi komentar YouTube berbahasa Indonesia menggunakan teknik Natural Language Processing.
2. Membandingkan performa tiga model — BiGRU, LSTM, dan IndoBERT — dalam mendeteksi komentar judi online.
3. Mengimplementasikan model terbaik ke dalam *browser extension* berbasis Manifest V3 untuk mendeteksi dan menyaring komentar judi online pada YouTube secara *real-time*.

---

# BAB II DATASET

## 2.1 Sumber Dataset

Dataset yang digunakan dalam penelitian ini merupakan gabungan dari enam dataset publik yang tersedia di platform Kaggle. Penggunaan beberapa dataset bertujuan untuk memperoleh variasi komentar yang lebih beragam sehingga model dapat mengenali berbagai pola komentar promosi judi online di media sosial, khususnya YouTube. Seluruh dataset berisi komentar berbahasa Indonesia yang telah diberi label atau dapat diproses menjadi dua kelas, yaitu *judol* dan *bukan_judol*. Penggabungan dataset juga dilakukan untuk memenuhi kebutuhan jumlah data minimal yang digunakan dalam proses pelatihan model.

Berikut adalah sumber dataset yang digunakan:

| No | Nama Dataset | Sumber (Kaggle) |
|:--:|--------------|-----------------|
| 1 | Judi Online | Yaemico |
| 2 | Deteksi Judi Online | Yaemico |
| 3 | Dataset Komentar Judi Online di YouTube | Kyyyy8 |
| 4 | Komentar Judi Online | Fahruu |
| 5 | Dataset Komentar Judi Online Platform YouTube | Kyyyy8 |
| 6 | Dataset Scraping Komentar Judi Online | Alexandro Kalindra |

## 2.2 Karakteristik Dataset

Setelah melalui tahap penggabungan, pembersihan, dan deduplikasi, diperoleh dataset akhir dengan karakteristik sebagai berikut:

| Metrik | Nilai |
|--------|-------|
| **Total baris** | 70.379 |
| **Label 0 (Bukan Judol)** | 62.202 (88,4%) |
| **Label 1 (Judol)** | 8.177 (11,6%) |
| **Kolom** | `text`, `label` |
| **Rata-rata panjang teks (kata)** | ~11 kata |
| **Persentil ke-95 panjang teks** | 31 kata |
| **Persentil ke-99 panjang teks** | 66 kata |

Dataset mengalami **ketidakseimbangan kelas (class imbalance)** yang signifikan dengan rasio sekitar **7,6:1** antara kelas bukan judol dan judol. Oleh karena itu, diperlukan teknik mitigasi seperti *class weights* atau *focal loss* pada saat pelatihan model.

**Ciri khas komentar judol:**
- Nama situs tersamar (contoh: *alexis17*, *pulauwin*, *sgi88*, *nyangkut77*, *probet855*)
- Kata promosi (contoh: *depo*, *wd*, *gacor*, *maxwin*, *cuan*, *bonus*, *jackpot*)
- Emoji tertentu setelah dikonversi (contoh: *red_heart*, *fire*, *star*)
- Pola kalimat yang cenderung seragam (*template-based*)

**Ciri khas komentar bukan judol:**
- Bahasa percakapan sehari-hari (contoh: *bisa*, *sudah*, *iya*, *kak*)
- Variasi panjang teks yang lebih lebar (standar deviasi 17 kata)
- Ekspresi emosi seperti *face_with_tears_of_joy* (emoji menangis)

## 2.3 Pembagian Data

Dataset dibagi menjadi data latih dan data uji dengan teknik *stratified sampling* untuk menjaga proporsi kelas. Detail pembagian untuk masing-masing model:

| Model | Rasio Split | Random State | Rincian |
|-------|-------------|:------------:|---------|
| **BiGRU** | 80/20 | 42 | Train: 56.303, Test: 14.076 |
| **LSTM** | 70/15/15 | 42 | Train: 49.265, Val: 7.038, Test: 14.076 |
| **IndoBERT-focal** | 80/20 | 42 | Train: 56.303, Test: 14.076 |

BiGRU dan IndoBERT-focal menggunakan split yang identik (80/20, *random_state*=42) sehingga perbandingan performa antar keduanya bersifat adil. LSTM menggunakan validation set terpisah dengan rasio 70/15/15, namun ukuran test set tetap sama (14.076 baris).

---

# BAB III PIPELINE NLP

Pipeline NLP pada proyek ini terdiri dari enam tahap utama yang dijalankan secara berurutan: pengumpulan dataset, preprocessing, tokenisasi, pelatihan model, evaluasi model, dan implementasi.

## 3.1 Pengumpulan Dataset

Tahap awal pipeline adalah mengumpulkan data komentar YouTube berbahasa Indonesia dari enam sumber dataset Kaggle. Dataset mentah digabungkan menggunakan *script* Python pada Notebook `DataPreparation.ipynb`. Proses penggabungan meliputi:

1. **Loading** — Membaca masing-masing file CSV menggunakan `pandas`.
2. **Standardisasi kolom** — Menyamakan nama dan format kolom menjadi `text` (isi komentar) dan `label` (0 untuk bukan judol, 1 untuk judol).
3. **Deduplikasi** — Menghapus baris duplikat berdasarkan teks komentar.
4. **Validasi label** — Memastikan hanya nilai 0 dan 1 yang terdapat pada kolom label.
5. **Output** — Menyimpan hasil ke `dataset_prepared.csv`.

Hasil akhir penggabungan adalah **70.379 baris data** yang siap diproses lebih lanjut.

## 3.2 Preprocessing

Preprocessing dilakukan melalui dua lapisan (*layer*) yang identik antara proses pelatihan (Python) dan runtime ekstensi (JavaScript), sehingga konsistensi data antara training dan inferensi tetap terjaga.

**Layer 1 — Advanced Preprocessing:**

| Langkah | Contoh Input | Output | Tujuan |
|---------|-------------|--------|--------|
| **Demojize** (emoji → teks) | 🥺 | `pleading_face` | Mengubah emoji menjadi deskripsi teks sebelum proses ASCII |
| **Unicode NFKD** | 𝘀𝗶𝘁𝘂𝘀 | `situs` | Menormalisasi font "fancy" Unicode ke bentuk dasar |
| **URL Masking** | `bit.ly/judi` | `[URL]` | Menyamarkan tautan dengan token khusus |
| **Lowercasing** | "BONUS DEPOSIT" | "bonus deposit" | Standarisasi kapitalisasi |
| **Symbol Cleaning** | `░▒▓█` | *(dihapus)* | Menghapus simbol non-alfanumerik |
| **Slang Normalisasi** (saka-nlp) | `wd`, `depo` | `withdraw`, `deposit` | Menormalkan singkatan dan bahasa tidak baku |
| **Repetisi Reduksi** | `bngettt` | `bngett` | Mengompres karakter berulang (maks 2 karakter) |
| **Whitespace Cleanup** | — | — | Menghapus spasi berlebih |

**Layer 2 — GRU-specific Cleaning (khusus BiGRU):**
Tambahan pembersihan untuk menyesuaikan dengan vocab model BiGRU: masking URL ulang, penghapusan simbol tersisa, dan normalisasi slang tambahan menggunakan kamus khusus (`SLANG_JUDOL_MAP`).

Pipeline preprocessing lengkap diimplementasikan dalam fungsi `final_preprocessing_pipeline()` pada Notebook `AdvancedPreprocessing.ipynb`. Di sisi ekstensi browser, pipeline yang sama diimplementasikan dalam `preprocess.js` menggunakan kamus emoji dan slang yang disalin dari Python.

## 3.3 Tokenisasi

Setelah preprocessing, teks dikonversi menjadi representasi numerik melalui tokenisasi. Pendekatan tokenisasi dibedakan berdasarkan jenis model:

**a) Word-Level Tokenization (untuk BiGRU dan LSTM)**

Kata-kata dalam teks dipetakan menjadi ID unik berdasarkan kamus (*vocabulary*) yang dibangun dari seluruh data latih.

| Parameter | BiGRU | LSTM |
|-----------|:-----:|:----:|
| **Framework** | PyTorch (kustom) | TensorFlow/Keras |
| **Vocab Size** | 23.154 kata | 15.000 kata |
| **Max Length** | 50 token | 100 token |
| **OOV Token** | `<UNK>` | `<OOV>` |
| **Padding** | Post-padding | Post-padding |

**b) Subword Tokenization (untuk IndoBERT)**

Menggunakan tokenizer bawaan dari model `indobenchmark/indobert-base-p1` dengan algoritma WordPiece (~32K vocabulary). Parameter:

| Parameter | Nilai |
|-----------|-------|
| **Tokenizer** | `AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")` |
| **Max Length** | 128 token |
| **Truncation** | Yes |
| **Padding** | Max length |

## 3.4 Pelatihan Model

Tiga model dilatih secara terpisah:

1. **BiGRU (PyTorch)** — dilatih pada GPU NVIDIA RTX 3050 6GB
2. **LSTM (TensorFlow/Keras)** — dilatih pada CPU
3. **IndoBERT-focal (HuggingFace Transformers)** — dilatih pada GPU dengan Focal Loss

Detail konfigurasi masing-masing model dijelaskan pada BAB IV.

## 3.5 Evaluasi Model

Setiap model dievaluasi menggunakan metrik:
- **Accuracy**, **Precision**, **Recall**, **F1-Score** (per kelas dan *macro/weighted average*)
- **Confusion Matrix** (True Positive, True Negative, False Positive, False Negative)
- **ROC-AUC** (khusus LSTM)
- **Error Analysis** — analisis manual terhadap sampel *false positive* dan *false negative*

Hasil evaluasi lengkap disajikan pada BAB VI.

## 3.6 Implementasi

Model dengan performa terbaik (IndoBERT-focal) dan alternatif ringan (BiGRU) dideploy sebagai layanan API menggunakan **FastAPI** pada platform **Hugging Face Spaces**. Browser extension pada Google Chrome/Edge (Manifest V3) mengirimkan komentar yang telah dipreprocessing ke API untuk mendapatkan prediksi, kemudian menerapkan aksi *Hide* atau *Highlight* sesuai pengaturan pengguna. Detail implementasi dijelaskan pada BAB V.

---

# BAB IV MODEL YANG DIGUNAKAN

## 4.1 Gated Recurrent Unit (GRU)

Gated Recurrent Unit (GRU) merupakan pengembangan dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah *vanishing gradient*. GRU menggunakan dua mekanisme — *update gate* dan *reset gate* — untuk mengatur informasi yang dipertahankan maupun diperbarui selama proses pembelajaran. Dibandingkan LSTM, GRU memiliki jumlah parameter yang lebih sedikit sehingga proses pelatihan lebih cepat dengan performa yang tetap baik pada tugas klasifikasi teks.

Pada penelitian ini, BiGRU (*Bidirectional GRU*) digunakan dengan konfigurasi:

| Parameter | Nilai |
|-----------|-------|
| **Framework** | PyTorch |
| **Arsitektur** | Bidirectional GRU (2 layer) |
| **Embedding Dim** | 128 |
| **Hidden Dim** | 64 |
| **Dropout** | 0,3 |
| **Loss Function** | BCEWithLogitsLoss |
| **Optimizer** | Adam (lr=0,001) |
| **Batch Size** | 64 |
| **Epochs** | 5 |
| **Device** | GPU (NVIDIA RTX 3050 6GB) |

BiGRU dilatih *from scratch* (tanpa *pre-trained weights*) dengan vocabulary 23.154 kata dan max length 50 token.

## 4.2 Long Short-Term Memory (LSTM)

Long Short-Term Memory (LSTM) merupakan pengembangan dari RNN yang memiliki tiga mekanisme gerbang — *forget gate*, *input gate*, dan *output gate* — untuk menyimpan informasi jangka panjang. LSTM banyak digunakan dalam berbagai tugas Natural Language Processing karena mampu mempelajari hubungan antar kata dalam suatu kalimat.

Pada penelitian ini, LSTM digunakan sebagai model pembanding dengan konfigurasi:

| Parameter | Nilai |
|-----------|-------|
| **Framework** | TensorFlow / Keras |
| **Arsitektur** | Bidirectional LSTM (2 layer: 64 → 32 units) |
| **Embedding Dim** | 128 |
| **Max Vocab** | 15.000 kata |
| **Max Length** | 100 token |
| **Loss Function** | Binary Crossentropy |
| **Optimizer** | Adam (lr=0,001) |
| **Class Weights** | {0: 0,567, 1: 4,303} |
| **Batch Size** | 128 |
| **Max Epochs** | 20 (Early Stopping, patience=4) |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau |
| **Device** | CPU |

LSTM menggunakan *class weights* untuk mengatasi ketidakseimbangan kelas, dengan bobot lebih besar pada kelas judol (1: 4,303).

## 4.3 IndoBERT

IndoBERT merupakan model berbasis arsitektur Transformer yang telah dilatih menggunakan korpus teks berbahasa Indonesia. Model ini memanfaatkan mekanisme *self-attention* sehingga mampu memahami konteks kalimat secara lebih baik dibandingkan model RNN.

Pada penelitian ini, IndoBERT digunakan melalui proses *fine-tuning* dengan konfigurasi:

| Parameter | Nilai |
|-----------|-------|
| **Framework** | HuggingFace Transformers |
| **Pre-trained Model** | `indobenchmark/indobert-base-p1` |
| **Max Length** | 128 token |
| **Loss Function** | Focal Loss (α=[1,0; 7,61], γ=2,0) |
| **Optimizer** | AdamW (lr=2e-5, weight_decay=0,01) |
| **Batch Size** | 32 (train) / 64 (eval) |
| **Epochs** | 5 |
| **FP16** | Enabled |
| **Callbacks** | EarlyStopping (patience=2) |
| **Device** | GPU (NVIDIA RTX 3050 6GB) |
| **Training Runtime** | ~52 menit (3.110 detik) |

**Focal Loss** dipilih untuk mengatasi ketidakseimbangan kelas. Berbeda dari *Cross-Entropy* standar, Focal Loss memodifikasi loss dengan faktor `(1 - pₜ)^γ` yang mengecilkan kontribusi loss dari sampel yang mudah diklasifikasi (kelas mayoritas) dan membesarkan loss dari sampel yang sulit (kelas minoritas judol). Bobot `alpha` memberikan penalti lebih besar untuk kesalahan pada kelas minoritas, dengan rasio 7,61:1 sesuai proporsi dataset.

---

# BAB V IMPLEMENTASI SISTEM

## 5.1 Arsitektur Sistem

Setelah proses pelatihan dan evaluasi model selesai dilakukan, dua model dideploy sebagai layanan API menggunakan FastAPI pada platform **Hugging Face Spaces**:

1. **IndoBERT-focal** (model utama — performa terbaik)
2. **BiGRU** (model alternatif — lebih ringan)

Browser extension menggunakan API endpoint untuk melakukan inferensi secara *online* (tidak lokal) sehingga tidak memerlukan sumber daya komputasi yang besar pada sisi klien. Arsitektur sistem secara keseluruhan digambarkan sebagai berikut:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ YouTube DOM │────>│  Content Script   │────>│   Service Worker     │
│  comments   │     │  (youtube.js)     │     │  (background.js)     │
└─────────────┘     │  + preprocess.js  │     └──────────┬───────────┘
        ▲           └──────────────────┘                │ POST /predict
        │                                                ▼
        │                                         ┌──────────────┐
        └─────────────────────────────────────────│ HF Spaces API │
                Hide / Highlight action            │  (FastAPI)    │
                                                  └──────────────┘
```

## 5.2 Alur Implementasi Sistem

Proses implementasi sistem berlangsung secara otomatis ketika pengguna membuka halaman video YouTube. Adapun alur implementasi sistem adalah sebagai berikut:

1. Pengguna mengaktifkan *browser extension* pada browser (Chrome/Edge).
2. Saat halaman YouTube dibuka, *content script* (`youtube.js`) memindai seluruh komentar menggunakan **MutationObserver** untuk menangani konten yang dimuat secara dinamis.
3. Setiap komentar yang ditemukan diproses menggunakan pipeline *preprocessing* JavaScript (`preprocess.js`) yang identik dengan pipeline preprocessing pada saat pelatihan model — meliputi normalisasi Unicode, normalisasi slang, pembersihan simbol, dan masking URL.
4. Hasil preprocessing dikirimkan secara *batch* ke **Service Worker**.
5. Service Worker mengirimkan permintaan **HTTP POST** ke API yang telah dideploy pada Hugging Face Spaces.
6. API menjalankan proses inferensi menggunakan model (IndoBERT-focal atau BiGRU) dan menghasilkan label prediksi beserta nilai *confidence*.
7. Browser extension menerima hasil prediksi, kemudian:
   - **Mode Hide** : Menyembunyikan komentar yang terdeteksi sebagai judol.
   - **Mode Highlight** : Memberikan penanda (border + label) beserta nilai *confidence* pada komentar yang terdeteksi.
8. Statistik jumlah komentar yang terdeteksi diperbarui secara otomatis pada popup extension.

## 5.3 Implementasi Browser Extension

Browser extension dikembangkan menggunakan **Chrome Extension Manifest V3** dengan bahasa pemrograman JavaScript. Extension terdiri atas beberapa komponen utama sebagai berikut:

| Komponen | File | Fungsi |
|----------|------|--------|
| **Content Script** | `content/youtube.js` | Mengambil komentar dari halaman YouTube via DOM dan menerapkan aksi hide/highlight |
| **Preprocessing Module** | `content/preprocess.js` | Melakukan preprocessing teks agar sesuai dengan pipeline pelatihan model |
| **Service Worker** | `background/background.js` | Menghubungkan extension dengan API model melalui HTTP Request |
| **Popup UI** | `popup/popup.html` + `popup.js` | Menampilkan pengaturan (Enable/Disable, Hide/Highlight, Threshold) serta statistik |
| **Styles** | `styles/` | CSS untuk efek hide dan highlight pada komentar |

Setiap komponen bekerja secara independen namun saling terintegrasi melalui *message passing* yang disediakan oleh Chrome Extension API.

## 5.4 Implementasi Antarmuka Browser Extension

Browser extension menyediakan antarmuka (*popup interface*) yang memudahkan pengguna dalam mengatur proses deteksi komentar judi online. Fitur-fitur yang tersedia meliputi:

1. **Enable/Disable Extension** — Tombol *master switch* untuk mengaktifkan atau menonaktifkan seluruh proses deteksi tanpa perlu melakukan uninstall.
2. **Mode Hide** — Menyembunyikan komentar yang terdeteksi sebagai promosi judi online dari tampilan halaman YouTube.
3. **Mode Highlight** — Memberikan penanda visual (border berwarna dan label "TERDETEKSI JUDOL" beserta nilai *confidence*) pada komentar yang terdeteksi tanpa menyembunyikannya.
4. **Confidence Threshold** — *Slider* untuk menentukan batas minimum tingkat keyakinan model (0,50–0,95) sebelum komentar dianggap sebagai judol.
5. **API Status** — Indikator yang menampilkan status koneksi antara browser extension dengan layanan API (terhubung/terputus).
6. **Statistik Deteksi** — Menampilkan jumlah komentar yang berhasil dideteksi pada halaman video yang sedang dibuka.

Melalui antarmuka tersebut, pengguna dapat mengatur perilaku browser extension sesuai dengan kebutuhan tanpa harus mengubah konfigurasi program secara langsung.

---

# BAB VI HASIL & EVALUASI

## 6.1 Hasil Implementasi Sistem

Setelah browser extension berhasil diimplementasikan, dilakukan pengujian pada beberapa halaman video YouTube yang mengandung komentar dengan karakteristik berbeda. Browser extension mampu mendeteksi komentar secara otomatis ketika komentar dimuat pada halaman tanpa memerlukan tindakan manual dari pengguna.

Salah satu hasil implementasi ditunjukkan pada gambar berikut:

![Hasil Deteksi Komentar](image)

Berdasarkan gambar tersebut, browser extension berhasil mengidentifikasi komentar yang mengandung unsur promosi judi online. Komentar tersebut mengandung beberapa kata kunci seperti *bonus deposit*, *Zeus*, dan *gampang menang* yang umum digunakan dalam promosi situs judi online. Model mengklasifikasikan komentar tersebut sebagai **TERDETEKSI JUDOL** dengan nilai *confidence* yang sangat tinggi.

## 6.2 Hasil Pengujian

Pengujian dilakukan untuk mengetahui kemampuan browser extension dalam mendeteksi komentar judi online secara otomatis pada halaman YouTube. Seluruh komentar yang muncul diproses menggunakan pipeline preprocessing sebelum dikirimkan ke API model.

Berdasarkan hasil pengujian, browser extension mampu:

- Mendeteksi komentar yang mengandung unsur promosi judi online secara otomatis.
- Menampilkan hasil prediksi beserta nilai *confidence*.
- Menjalankan aksi *Hide* maupun *Highlight* sesuai pengaturan pengguna.
- Memproses komentar yang baru dimuat (*dynamic loading*) tanpa perlu melakukan *refresh* halaman.

Selama pengujian, browser extension dapat bekerja dengan baik tanpa mengganggu proses pemuatan halaman YouTube. Penggunaan *MutationObserver* memungkinkan deteksi berjalan secara *real-time* saat pengguna melakukan *scroll* pada halaman komentar.

## 6.3 Evaluasi Model

Evaluasi dilakukan terhadap ketiga model yang dilatih — BiGRU, LSTM, dan IndoBERT-focal — pada dataset uji yang identik (14.076 sampel) untuk memperoleh perbandingan yang adil.

### Ringkasan Metrik

| Metrik | BiGRU | LSTM | 🏆 IndoBERT-focal |
|--------|:-----:|:----:|:-----------------:|
| **Accuracy** | 99,23% | 99% | **99,64%** |
| **Judol Precision** | 0,97 | 0,93 | **0,99** |
| **Judol Recall** | 0,96 | 0,97 | **0,98** |
| **Judol F1-Score** | 0,97 | 0,95 | **0,98** |
| **False Positive** | 49 | 119 | **23** |
| **False Negative** | 65 | 49 | **28** |
| **ROC-AUC** | — | 0,9953 | — |

### Perbandingan Arsitektur Model

| Aspek | BiGRU | LSTM | IndoBERT-focal |
|-------|:-----:|:----:|:--------------:|
| **Jenis Model** | RNN (GRU) | RNN (LSTM) | Transformer (BERT) |
| **Framework** | PyTorch | TensorFlow/Keras | HuggingFace Transformers |
| **Pre-trained** | ❌ (from scratch) | ❌ (from scratch) | ✅ indobert-base-p1 |
| **Tokenisasi** | Word-level (23K vocab) | Word-level (15K vocab) | Subword WordPiece (32K) |
| **Penanganan Imbalance** | — | Class Weights | **Focal Loss** (α=7,61, γ=2,0) |
| **Epochs** | 5 | 6 (early stop) | 5 |

### Analisis Per Model

**IndoBERT-focal** mencapai performa tertinggi dengan accuracy **99,64%** — hanya **23 False Positive** dan **28 False Negative** dari 14.076 sampel uji. Focal Loss dengan α=7,61 dan γ=2,0 terbukti efektif mengatasi ketidakseimbangan kelas. Analisis error menunjukkan bahwa sebagian besar kesalahan model kemungkinan disebabkan oleh *noise label dataset*, bukan kelemahan model.

**BiGRU** mencapai accuracy **99,23%** dengan hanya 49 False Positive. Model ini jauh lebih ringan dari IndoBERT (parameter lebih sedikit), menjadikannya alternatif yang cocok untuk lingkungan dengan sumber daya terbatas.

**LSTM** memiliki recall Judol tertinggi (0,97) setelah IndoBERT, dengan ROC-AUC **0,9953** yang menunjukkan kemampuan diskriminasi kelas yang sangat baik. Namun, precision Judol lebih rendah (0,93) dengan 119 False Positive — tertinggi di antara ketiga model.

### Confusion Matrix

| Model | | Prediksi: Bukan Judol | Prediksi: Judol |
|-------|----------------------|:---------------------:|:---------------:|
| **BiGRU** | Aktual: Bukan Judol | 12.392 (TN) | 49 (FP) |
| | Aktual: Judol | 65 (FN) | 1.570 (TP) |
| **LSTM** | Aktual: Bukan Judol | 12.322 (TN) | 119 (FP) |
| | Aktual: Judol | 49 (FN) | 1.586 (TP) |
| **IndoBERT** | Aktual: Bukan Judol | 12.418 (TN) | **23 (FP)** |
| | Aktual: Judol | **28 (FN)** | 1.607 (TP) |

### Insight Utama

1. **IndoBERT-focal adalah model terbaik** — accuracy 99,64%, hanya 23 FP & 28 FN. Analisis error menunjukkan sebagian besar kesalahan adalah *noise label dataset*, bukan kelemahan model.
2. **Focal Loss terbukti efektif** untuk mengatasi ketidakseimbangan kelas pada dataset judol. Kombinasi α=7,61 dan γ=2,0 memungkinkan model fokus pada sampel kelas minoritas tanpa mengorbankan spesifisitas.
3. **BiGRU alternatif ringan yang solid** — accuracy 99,23%, ukuran model jauh lebih kecil, cocok untuk inferensi *real-time* di lingkungan dengan resource terbatas.
4. **Pre-trained Transformer (IndoBERT) mengalahkan RNN from scratch** — temuan ini konsisten dengan literatur NLP bahwa *fine-tuning* model pre-trained memberikan hasil lebih baik, terutama setelah stabilitas training terkendali.

## 6.4 Analisis Hasil

Hasil implementasi menunjukkan bahwa sistem mampu menjalankan fungsi utama sebagai sistem penyaring komentar judi online pada platform YouTube.

**Faktor-faktor yang mendukung keberhasilan deteksi:**

1. **Dataset yang beragam** — Penggabungan enam sumber dataset berbeda memungkinkan model mempelajari variasi komentar yang lebih luas.
2. **Pipeline preprocessing yang komprehensif** — Mampu menormalkan emoji, Unicode, slang, URL, dan karakter berulang sehingga komentar memiliki bentuk yang lebih konsisten sebelum diproses model.
3. **Konsistensi preprocessing** — Pipeline preprocessing yang identik antara pelatihan (Python) dan runtime ekstensi (JavaScript) menjaga konsistensi hasil prediksi.
4. **Focal Loss pada IndoBERT** — Berhasil mengatasi ketidakseimbangan kelas dengan menekan loss dari sampel mudah dan memperbesar loss dari sampel sulit.

**Keterbatasan sistem:**

1. Koneksi internet diperlukan — ekstensi bergantung pada API Hugging Face Spaces untuk inferensi.
2. Komentar dengan istilah baru atau penyamaran kata yang belum terdapat pada kamus normalisasi berpotensi menurunkan akurasi prediksi.
3. Penggunaan *font* Unicode yang sangat ekstrem atau kombinasi simbol yang kompleks masih dapat lolos dari preprocessing.
4. Ketergantungan pada struktur DOM YouTube yang dapat berubah sewaktu-waktu.

---

# BAB VII KESIMPULAN

## 7.1 Kesimpulan

Berdasarkan hasil penelitian dan implementasi yang telah dilakukan, dapat disimpulkan beberapa hal sebagai berikut:

1. **Sistem pendeteksi komentar judi online** berbasis Natural Language Processing berhasil dikembangkan dalam bentuk browser extension (Chrome/Edge, Manifest V3) yang dapat mendeteksi komentar promosi judi online pada YouTube secara otomatis melalui tahapan preprocessing, klasifikasi menggunakan model NLP, serta penerapan aksi Hide (menyembunyikan) atau Highlight (menandai) sesuai pengaturan pengguna.

2. **Perbandingan tiga model klasifikasi** menunjukkan bahwa **IndoBERT-focal** memberikan performa terbaik secara keseluruhan dengan accuracy **99,64%**, Judol Precision **0,99**, Judol Recall **0,98**, dan hanya **23 False Positive** serta **28 False Negative** dari 14.076 sampel uji. Model BiGRU menyusul dengan accuracy 99,23% (49 FP, 65 FN), dan LSTM dengan accuracy 99% (119 FP, 49 FN, ROC-AUC 0,9953).

3. **Focal Loss terbukti efektif** dalam mengatasi ketidakseimbangan kelas (rasio ~7,6:1). Dengan α=7,61 dan γ=2,0, IndoBERT-focal mampu memfokuskan pembelajaran pada sampel kelas minoritas tanpa mengorbankan spesifisitas terhadap kelas mayoritas.

4. **Implementasi browser extension** berhasil membuktikan bahwa model NLP yang telah dilatih dapat diterapkan pada lingkungan penggunaan nyata. Pengujian menunjukkan bahwa ekstensi mampu mendeteksi komentar judi online dengan tingkat keyakinan tinggi secara *real-time* tanpa mengganggu pengalaman pengguna.

5. **Konsistensi pipeline preprocessing** antara Python (training) dan JavaScript (runtime) menjadi faktor kunci dalam menjaga akurasi prediksi pada saat inferensi.

## 7.2 Saran

Meskipun sistem telah berhasil diimplementasikan dan menunjukkan performa yang baik, masih terdapat beberapa aspek yang dapat dikembangkan pada penelitian selanjutnya:

1. **Dataset** — Menambahkan dataset yang lebih beragam dan lebih mutakhir agar model mampu mengenali pola komentar judi online terbaru yang terus berkembang. Review label pada 51 sampel error (23 FP + 28 FN) juga direkomendasikan untuk memvalidasi *noise label* dan meningkatkan akurasi model lebih lanjut.

2. **Ketahanan model** — Mengembangkan model agar lebih tahan terhadap teknik penyamaran kata, penggunaan simbol, karakter Unicode, maupun kombinasi emoji yang sering digunakan untuk menghindari deteksi.

3. **Optimasi performa** — Mengoptimalkan kecepatan inferensi browser extension, terutama saat memproses komentar dalam jumlah besar. Opsi yang dapat dipertimbangkan meliputi penggunaan ONNX Runtime di browser atau model yang lebih ringan (BiGRU) untuk skenario *offline*.

4. **Fitur pelaporan** — Menambahkan fitur pelaporan (*report*) komentar yang terdeteksi sehingga pengguna dapat langsung melaporkan komentar tersebut kepada platform YouTube.

5. **Deteksi multimodal** — Mengembangkan sistem agar tidak hanya mendeteksi komentar berbentuk teks, tetapi juga mampu menganalisis konten lain seperti gambar profil atau tautan yang berpotensi digunakan sebagai media promosi judi online.

6. **Pembaruan berkala** — Melakukan pelatihan ulang model secara berkala dengan data baru agar sistem tetap adaptif terhadap perubahan pola promosi judi online.

---

# DAFTAR PUSTAKA

1. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *arXiv preprint arXiv:1406.1078*.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

4. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

5. Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP. *Proceedings of the 28th International Conference on Computational Linguistics*, 557–570.

6. TensorFlow Developers. (2024). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. *arXiv preprint arXiv:1603.04467*.

7. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32*, 8024–8035.

8. Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45.

9. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

10. Saka NLP. (2024). saka: Indonesian Text Normalization Library. *Python Package Index (PyPI)*.