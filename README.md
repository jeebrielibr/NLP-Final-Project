# YouTube Judol Comment Remover

Ekstensi browser berbasis NLP untuk **mendeteksi**, **menyembunyikan**, dan **menyorot** komentar judi online (*judol*) di YouTube — proyek final mata kuliah Natural Language Processing.

---

## Informasi Proyek

| Item | Keterangan |
|------|------------|
| **Kelompok** | Newton |
| **Program Studi** | Teknik Informatika |
| **Mata Kuliah** | Natural Language Processing |
| **Dosen Pengampu** | Muhammad Yazid Supriadi S.Kom, M.Kom |
| **Asisten Dosen/Lab** | Romi Wahyudi |
| **Domain** | Sosial Media |
| **Jenis Aplikasi** | Text Classification + Browser Extension |
| **Bahasa** | Indonesia |

### Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1 | Muhammad Jibril Ibrahim | 0110224002 |
| 2 | Rohmatul Hidayat | 0110224015 |
| 3 | Achmad Muflih Alrasyid | 0110224162 |
| 4 | Muhammad Ridwan Karim | 0110224122 |
| 5 | Anwar Maulana | 0110224020 |

---

## Latar Belakang

Komentar judi online (*judol*) di YouTube semakin marak — sering berupa promosi situs judi, link referral, atau ajakan bermain dengan iming-iming keuntungan cepat. Konten ini mengganggu pengalaman menonton, berpotensi menyesatkan penonton (terutama remaja), dan sulit diblokir secara manual karena muncul dalam variasi teks yang terus berubah.

**Masalah yang diselesaikan:**

- Pengguna YouTube kesulitan memfilter komentar judol secara manual
- Pola teks judol bervariasi (slang, typo, emoji, link tersembunyi)
- Moderasi platform tidak selalu cukup cepat atau konsisten

**Solusi:** Ekstensi browser yang secara otomatis mendeteksi komentar judol menggunakan model NLP berbahasa Indonesia, lalu menyembunyikan atau menyorot komentar tersebut agar pengguna dapat membuktikan bahwa sistem bekerja.

---

## Fitur Utama

### Model NLP

- Klasifikasi biner: `judol` vs `bukan_judol`
- Tiga model dilatih: **BiGRU** (PyTorch), **LSTM** (TensorFlow), **IndoBERT + Focal Loss** (HuggingFace)
- Preprocessing teks Indonesia informal: emoji → teks, normalisasi Unicode (bold/italic), masking URL, normalisasi slang (saka-nlp), reduksi repetisi karakter
- Evaluasi: accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC

### Ekstensi Browser (Chrome/Edge)

| Fitur | Deskripsi |
|-------|-----------|
| **Hide** | Menyembunyikan komentar terdeteksi judol |
| **Highlight** | Menyorot komentar dengan border + label confidence |
| Toggle mode | Pengguna memilih hide atau highlight |
| Threshold confidence | Filter prediksi dengan skor minimum (0,50–0,95) |
| Statistik sesi | Jumlah komentar terdeteksi per halaman video |
| Enable / disable | Master switch tanpa uninstall |
| API status | Indikator ketersediaan API di popup |

---

## Arsitektur Sistem

```
[Komentar YouTube] --> [Content Script] --> [Preprocessing JS] --> [Service Worker]
                         │                                            │
                         │ DOM observer +                         fetch POST /predict
                         │ batch queue                              (HF Spaces API)
                         │                                            │
                         └──────────── [Action: Hide/Highlight] ←───┘
```

### Alur Runtime (saat pengguna membuka YouTube)

1. Content script (`youtube.js`) memindai komentar via `MutationObserver` pada `ytd-comment-thread-renderer`
2. Teks komentar dipreprocess di sisi klien (`preprocess.js`) — mirror pipeline Python
3. Komentar dikirim batch ke service worker → POST ke Hugging Face Space API
4. API menjalankan inferensi IndoBERT → mengembalikan `{label, score}`
5. Content script menerapkan aksi sesuai pengaturan: **Hide** atau **Highlight** + badge

### Alur Offline (pelatihan model)

1. **Fetch data** — 6 dataset dari Kaggle (komentar judi online YouTube)
2. **Preprocessing** — emoji demojize, Unicode NFKD, URL masking, saka-nlp slang normalization
3. **Labeling** — anotasi: `0` (bukan_judol) / `1` (judol)
4. **Training** — BiGRU + LSTM (baseline) + IndoBERT fine-tune dengan Focal Loss
5. **Evaluasi** — classification report, confusion matrix, error analysis (FP/FN)
6. **Export** — model disimpan ke `model/`, API di-deploy ke Hugging Face Spaces

---

## Pipeline NLP

Pipeline preprocessing diterapkan secara identik di Python (training) dan JavaScript (runtime ekstensi):

| Langkah | Keterangan |
|---------|-----------|
| **Emoji conversion** | `emoji.demojize()` → teks (mis. 🥺 → `pleading_face`) |
| **Unicode normalization** | NFKD — bold/italic/fractur → normal ASCII |
| **URL masking** | URL, bit.ly, `[dot]` obfuscation → `[URL]` |
| **Lowercasing** | Standarisasi huruf |
| **Symbol cleaning** | Hanya alfanumerik + `[URL]` + underscore/colon |
| **Slang normalization** | saka-nlp + kamus judol kustom (mis. "wd" → "withdraw", "depo" → "deposit") |
| **Char repetition** | "bngettt" → "bngett" (3+ → 2) |
| **Whitespace cleanup** | Normalisasi spasi |

---

## Model & Hasil Evaluasi

Tiga model dilatih pada dataset yang sama (70.379 baris). Detail lengkap ada di [Laporan Evaluasi Model](Docs/Laporan_Evaluasi_Model.md).

### Ringkasan Metrik

| Metrik | BiGRU | LSTM | IndoBERT-focal |
|--------|:-----:|:----:|:--------------:|
| **Accuracy** | 99,23% | 99% | 52% |
| **Judol Precision** | 0,97 | 0,93 | 0,14 |
| **Judol Recall** | 0,96 | 0,97 | 0,62 |
| **Judol F1-Score** | 0,97 | 0,95 | 0,23 |
| **False Positive** | 49 | 119 | 6.169 |
| **False Negative** | 65 | 49 | 614 |
| **ROC-AUC** | — | 0,9953 | — |

### Karakteristik Model

| Aspek | BiGRU | LSTM | IndoBERT-focal |
|-------|-------|------|----------------|
| **Jenis** | RNN (GRU) | RNN (LSTM) | Transformer (BERT) |
| **Framework** | PyTorch | TensorFlow/Keras | HuggingFace Transformers |
| **Pre-trained** | Tidak | Tidak | indobert-base-p1 |
| **Tokenisasi** | Word-level | Word-level | Subword (WordPiece) |
| **Split** | 80/20 | 70/15/15 | 80/20 |
| **Penanganan Imbalance** | — | Class Weights | Focal Loss (α=7,61, γ=2,0) |

**Kesimpulan:** BiGRU menjadi model terbaik dengan accuracy 99,23% dan Judol F1 0,97. IndoBERT-focal mengalami kegagalan generalisasi (training loss 0,0195 namun test accuracy 52%) akibat Focal Loss yang terlalu agresif.

---

## Dataset

| Aspek | Detail |
|-------|--------|
| **Total baris** | 70.379 |
| **Label 0 (Bukan Judol)** | 62.202 (88,4%) |
| **Label 1 (Judol)** | 8.177 (11,6%) |
| **Kolom** | `text`, `label` |
| **Sumber** | 6 dataset Kaggle (lihat [Dataset/Notes/Sumber.txt](Dataset/Notes/Sumber.txt)) |

**Karakteristik judol:** nama situs tersamar (`alexis17`, `pulauwin`, `sgi88`), kata promosi (`depo`, `wd`, `gacor`, `maxwin`), emoji tertentu (`red_heart`, `fire`, `star`).

---

## Tech Stack

| Lapisan | Teknologi |
|---------|-----------|
| **Training** | Python 3.10+, PyTorch, TensorFlow/Keras, HuggingFace Transformers |
| **Data & evaluasi** | pandas, scikit-learn, matplotlib, seaborn, saka-nlp |
| **Deploy model** | Hugging Face Spaces (FastAPI + Docker) |
| **Ekstensi** | Chrome Extension Manifest V3 (JavaScript) |
| **Preprocessing JS** | Custom pipeline (emoji map, Unicode normalization, slang dictionary) |

---

## Struktur Repository

```
NLP Final Project/
├── Dataset/                     # Dataset mentah & terproses
│   ├── datasetraw1–6.csv        # 6 dataset mentah dari Kaggle
│   ├── dataset_prepared.csv     # Hasil merge (±70K baris)
│   ├── dataset_clean_final.csv  # Dataset final siap training (70.379 baris)
│   └── Notes/Sumber.txt         # Link sumber dataset Kaggle
├── Notebook/                    # Jupyter notebooks
│   ├── DataPreparation.ipynb    # Merge & deduplication
│   ├── EDA.ipynb                # Exploratory data analysis
│   ├── AdvancedPreprocessing.ipynb  # Pipeline preprocessing lanjutan
│   ├── FeatureExtraction.ipynb  # Tokenisasi word-level & subword
│   ├── GRU_model.ipynb          # Training BiGRU (PyTorch)
│   ├── LSTM.ipynb               # Training BiLSTM (TensorFlow)
│   ├── IndoBERT_focal.ipynb     # Fine-tune IndoBERT + Focal Loss
│   └── archive/                 # Notebook lama (IndoBERT non-focal)
├── model/                       # Model tersimpan
│   ├── bigru/                   # bigru_model.pt + bigru_config.pt
│   ├── indobert_judol_model/    # IndoBERT non-focal (deprecated)
│   ├── indobert_judol_model_focal/  # IndoBERT + Focal Loss
│   └── tokenizer_bigru.pickle
├── deployhf/                    # Deploy Hugging Face Spaces
│   ├── app.py                   # FastAPI endpoint /predict
│   ├── Dockerfile               # Python 3.10-slim, port 7860
│   └── requirements.txt
├── extension/                   # Chrome/Edge Extension (Manifest V3)
│   ├── manifest.json
│   ├── background/service-worker.js   # API calls + message routing
│   ├── content/youtube.js             # DOM observer + hide/highlight
│   ├── content/preprocess.js          # JS preprocessing pipeline
│   ├── popup/popup.{html,js,css}      # Settings UI
│   ├── styles/highlight.css           # Highlight/hide CSS
│   └── assets/                        # Icons (16/48/128px)
├── Docs/                        # Dokumentasi
│   ├── Desc.md                  # Ketentuan mata kuliah
│   ├── Rencana_Proyek.md        # Rencana detail proyek
│   ├── Rencana_Data_Preparation.md
│   ├── EDA_Report.md            # Laporan EDA
│   ├── Advanced_Preprocessing_Strategy.md
│   ├── Feature_Extraction_Pipeline.md
│   ├── Laporan_Evaluasi_Model.md  # Laporan hasil evaluasi 3 model
│   └── sketsa_projek.png
├── Laporan/                     # File laporan final (PDF)
├── .gitignore
└── README.md
```

---

## Deployment

### Hugging Face Spaces

API berjalan di Hugging Face Spaces menggunakan FastAPI + Docker:

| Komponen | Detail |
|----------|--------|
| **Endpoint** | `POST /predict` |
| **Input** | `{"text": "komentar youtube"}` |
| **Output** | `{"label": "judol"|"bukan_judol", "score": 0.98}` |
| **Port** | 7860 (standar HF Spaces) |
| **Model** | IndoBERT + Focal Loss |

> **Catatan:** Model yang saat ini di-deploy adalah IndoBERT-focal. Berdasarkan hasil evaluasi, disarankan untuk mengganti dengan BiGRU yang memiliki performa jauh lebih baik. Lihat [Laporan Evaluasi](Docs/Laporan_Evaluasi_Model.md).

### Browser Extension

1. Buka `chrome://extensions` (atau `edge://extensions`)
2. Aktifkan **Developer Mode**
3. Klik **Load unpacked** → pilih folder `extension/`
4. Buka video YouTube — ekstensi otomatis memindai komentar
5. Klik icon ekstensi untuk mengatur mode (hide/highlight), threshold, dan melihat statistik

---

## Dokumentasi

| Dokumen | Isi |
|---------|-----|
| [Rencana Proyek](Docs/Rencana_Proyek.md) | Arsitektur, pipeline, evaluasi, rencana implementasi |
| [Ketentuan Mata Kuliah](Docs/Desc.md) | Informasi kelompok, syarat, bobot penilaian |
| [Laporan EDA](Docs/EDA_Report.md) | Analisis distribusi label, panjang teks, kata kunci |
| [Strategi Preprocessing](Docs/Advanced_Preprocessing_Strategy.md) | Detail pipeline preprocessing lanjutan |
| [Pipeline Feature Extraction](Docs/Feature_Extraction_Pipeline.md) | Tokenisasi word-level & subword |
| [Laporan Evaluasi Model](Docs/Laporan_Evaluasi_Model.md) | Hasil evaluasi BiGRU, LSTM, IndoBERT-focal |

---

## Ketentuan Mata Kuliah

Proyek ini memenuhi ketentuan final project NLP:

- **Domain:** Sosial Media (YouTube)
- **Task:** Text Classification (`judol` / `bukan_judol`)
- **Teknologi:** BiGRU/LSTM (wajib) + IndoBERT/HuggingFace — ✅ 3 model dilatih
- **Dataset:** ≥ 10.000 komentar berbahasa Indonesia — ✅ 70.379 baris
- **Deploy:** Minimal Hugging Face Spaces + demo ekstensi browser — ✅ keduanya aktif

Detail ketentuan dan bobot penilaian ada di [`Docs/Desc.md`](Docs/Desc.md).

---

## Deliverables

- [x] Repository kode lengkap
- [x] Model + demo Hugging Face Spaces
- [x] Ekstensi browser (hide + highlight)
- [x] Laporan evaluasi model ([Docs/Laporan_Evaluasi_Model.md](Docs/Laporan_Evaluasi_Model.md))
- [ ] Laporan PDF final
- [ ] Slide presentasi

---

## Lisensi

Proyek akademik — Teknik Informatika, mata kuliah Natural Language Processing.
