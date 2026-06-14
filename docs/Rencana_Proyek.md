# Rencana Proyek Final — Browser Extension Deteksi Komentar Judol di YouTube

**Judul kerja:** *YouTube Judol Comment Remover*  
**Domain:** Sosial Media  
**Jenis aplikasi:** Text Classification + Browser Extension  
**Bahasa:** Indonesia  
**Versi dokumen:** 1.0  
**Tanggal:** Juni 2026

---

## 1. Latar Belakang & Masalah

Komentar judi online (*judol*) di YouTube semakin marak — sering berupa promosi situs judi, link referral, atau ajakan bermain dengan iming-iming keuntungan cepat. Konten ini mengganggu pengalaman menonton, berpotensi menyesatkan penonton (terutama remaja), dan sulit diblokir secara manual karena muncul dalam variasi teks yang terus berubah.

**Masalah yang ingin diselesaikan:**
- Pengguna YouTube kesulitan memfilter komentar judol secara manual.
- Pola teks judol bervariasi (slang, typo, emoji, link tersembunyi).
- Moderasi platform tidak selalu cukup cepat atau konsisten.

**Solusi yang diusulkan:**  
Ekstensi browser yang secara otomatis **mendeteksi** komentar judol menggunakan model NLP berbahasa Indonesia, lalu **menyembunyikan** atau **menyorot (highlight)** komentar tersebut agar pengguna dapat membuktikan bahwa sistem bekerja.

---

## 2. Tujuan Proyek

| # | Tujuan | Indikator keberhasilan |
|---|--------|------------------------|
| 1 | Mendeteksi komentar judol secara otomatis | Model klasifikasi berjalan end-to-end |
| 2 | Mengurangi paparan komentar judol di YouTube | Mode hide aktif di ekstensi |
| 3 | Membuktikan deteksi berhasil | Mode highlight menampilkan label/kelas |
| 4 | Memenuhi syarat mata kuliah NLP | Pipeline NLP jelas, evaluasi terukur, deploy tersedia |

---

## 3. Ruang Lingkup

### Dalam ruang lingkup
- Pengumpulan dataset komentar YouTube berbahasa Indonesia
- Preprocessing dan pelabelan data (judol / bukan judol)
- Pelatihan dan evaluasi model NLP (min. 2 teknologi)
- Inferensi model untuk klasifikasi real-time atau semi-real-time
- Ekstensi browser (Chrome/Edge) untuk YouTube
- Mode **Hide** dan **Highlight** komentar terdeteksi
- Demo deploy model (Hugging Face) + dokumentasi

### Di luar ruang lingkup (fase awal)
- Moderasi otomatis di sisi server YouTube
- Deteksi konten video (hanya fokus komentar)
- Dukungan multi-platform selain YouTube
- Pelaporan komentar ke YouTube secara otomatis

---

## 4. Arsitektur Sistem

```
[Komentar YouTube] --> [Pipeline NLP] --> [Ekstensi Browser]
                         |
                         +-- fetch data (API)
                         +-- preprocessing
                         +-- training / inferensi
                         +-- output: judol / bukan_judol
                                              |
                                              +-- hide comments
                                              +-- highlight comments
```

### 4.1 Alur data (runtime — saat pengguna membuka YouTube)

1. Ekstensi membaca komentar dari halaman YouTube (Content Script / DOM observer).
2. Teks komentar dikirim ke modul inferensi (lokal via ONNX/Web, atau API Hugging Face).
3. Model mengembalikan prediksi: `judol` atau `bukan_judol`.
4. Ekstensi menerapkan aksi sesuai pengaturan pengguna:
   - **Hide:** komentar disembunyikan dari tampilan.
   - **Highlight:** komentar diberi border/warna + label (mis. "Terdeteksi Judol").

### 4.2 Alur data (offline — pelatihan model)

1. **Fetch data** — komentar YouTube via YouTube Data API v3 (untuk dataset).
2. **Preprocessing** — cleaning, normalisasi slang, tokenisasi, handling URL/emoji.
3. **Labeling** — anotasi manual: `judol` vs `bukan_judol`.
4. **Training** — baseline LSTM/GRU + model Transformer (IndoBERT/mBERT).
5. **Evaluasi** — metrik klasifikasi + analisis error.
6. **Export** — simpan model ke format deploy (Hugging Face Inference API atau ONNX).

---

## 5. Pipeline NLP

### 5.1 Preprocessing

| Langkah | Keterangan |
|---------|------------|
| Lowercasing / normalisasi | Standarisasi teks Indonesia informal |
| URL dan link normalization | Pola judol sering mengandung link |
| Emoji handling | Pertahankan atau normalisasi sebagai fitur |
| Slang dan typo mapping | Kamus slang judol (mis. "wd", "depo", "slot gacor") |
| Tokenisasi | Word-level (LSTM) dan subword/BPE (Transformer) |
| Padding / truncation | Panjang maksimum token disesuaikan distribusi komentar |

### 5.2 Model (memenuhi syarat min. 2 teknologi)

| Model | Peran | Alasan pemilihan |
|-------|-------|------------------|
| **BiGRU / LSTM** *(wajib)* | Baseline sequence model | Ringan, mudah dijelaskan alur embedding → RNN → dense |
| **IndoBERT / mBERT** (HuggingFace) | Model utama performa tinggi | Kuat untuk konteks bahasa Indonesia dan variasi teks |

**Strategi eksperimen:**
- Latih BiGRU sebagai baseline.
- Fine-tune IndoBERT pada dataset yang sama.
- Bandingkan F1, precision, recall, confusion matrix.
- Pilih model terbaik untuk inferensi di ekstensi (pertimbangkan ukuran dan latency).

### 5.3 Inferensi di ekstensi

| Opsi | Kelebihan | Kekurangan |
|------|-----------|------------|
| **A. Hugging Face Inference API** | Mudah deploy, tidak perlu GPU di client | Butuh internet, latency API |
| **B. ONNX Runtime di browser** | Offline-capable, lebih cepat setelah load | Setup konversi model lebih rumit |
| **C. Model ringan (TF.js / WASM)** | Cocok untuk demo lokal | Akurasi mungkin turun vs IndoBERT penuh |

**Rekomendasi awal:** IndoBERT di Hugging Face untuk demo dan evaluasi; BiGRU diekspor ringan untuk perbandingan latency di laporan.

---

## 6. Komponen Ekstensi Browser

### 6.1 Struktur folder (rencana)

```
extension/
├── manifest.json
├── background/service-worker.js
├── content/youtube.js
├── popup/popup.html
├── popup/popup.js
├── styles/highlight.css
└── lib/inference.js
```

### 6.2 Fitur ekstensi

| Fitur | Deskripsi |
|-------|-----------|
| Toggle Hide / Highlight | Pengguna memilih mode tampilan |
| Threshold confidence | Filter prediksi dengan skor minimum |
| Statistik sesi | Jumlah komentar terdeteksi per halaman video |
| Enable / disable | Master switch tanpa uninstall |

### 6.3 Integrasi YouTube

- **Runtime:** baca komentar dari DOM halaman watch YouTube (MutationObserver untuk lazy-load).
- **Dataset:** YouTube Data API v3 (`commentThreads.list`) untuk pengumpulan data berlabel.

---

## 7. Dataset

### 7.1 Kebutuhan

- **Minimal 10.000 data** berbahasa Indonesia (sesuai ketentuan mata kuliah).

### 7.2 Sumber data

| Sumber | Kegunaan |
|--------|----------|
| YouTube Data API v3 | Komentar dari video populer / trending |
| Dataset publik komentar ID (jika ada) | Data negatif (bukan judol) |
| Anotasi manual kelompok | Label `judol` / `bukan_judol` |

### 7.3 Skema label

| Label | Definisi |
|-------|----------|
| `judol` | Promosi judi online, link situs judi, ajakan deposit/main slot/casino |
| `bukan_judol` | Komentar normal, feedback video, diskusi umum |

### 7.4 Pembagian data

- Train: 70% | Validation: 15% | Test: 15%

---

## 8. Evaluasi dan Eksperimen

### 8.1 Metrik wajib

- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Analisis false positive / false negative

### 8.2 Eksperimen tambahan (nilai plus)

- Perbandingan BiGRU vs IndoBERT
- Dampak preprocessing (dengan/tanpa normalisasi slang)
- Analisis contoh komentar yang salah klasifikasi
- Pengukuran latency inferensi di ekstensi

### 8.3 Uji sistem (end-to-end)

| Skenario uji | Ekspektasi |
|--------------|------------|
| Video dengan komentar judol | Komentar terdeteksi dan di-hide/highlight |
| Komentar normal | Tidak ter-hide, tidak salah highlight |
| Toggle mode Hide ↔ Highlight | Perilaku UI berubah sesuai setting |
| Video tanpa komentar judol | Ekstensi tidak error |

---

## 9. Tech Stack

| Lapisan | Teknologi |
|---------|-----------|
| Training | Python 3.10+, PyTorch, HuggingFace Transformers |
| Baseline RNN | BiGRU/LSTM (PyTorch atau Keras) |
| Data & evaluasi | pandas, scikit-learn, matplotlib, seaborn |
| API dataset | YouTube Data API v3 |
| Deploy model | Hugging Face Spaces |
| Ekstensi | Chrome Extension Manifest V3 (JavaScript) |

---

## 10. Struktur Repository (rencana)

```
NLP Final Project/
├── docs/
│   ├── Desc.md
│   └── Rencana_Proyek.md
├── data/raw/
├── data/processed/
├── notebooks/
├── src/
├── extension/
├── deployment/huggingface/
├── reports/
├── requirements.txt
└── README.md
```

---

## 11. Deployment dan Demo

### Hugging Face (wajib minimal)
- Upload model terbaik ke Hugging Face Hub.
- Gradio Space: input teks komentar → output prediksi + confidence.

### Ekstensi browser
- Load unpacked di Chrome/Edge untuk demo presentasi.

### Checklist deliverables
- [ ] Repository kode lengkap
- [ ] Model + demo Hugging Face
- [ ] Ekstensi browser (hide + highlight)
- [ ] Laporan PDF 5–10 halaman
- [ ] Slide presentasi

---

## 12. Timeline Kerja (6–8 minggu)

| Minggu | Kegiatan | Output |
|--------|----------|--------|
| 1 | Finalisasi rencana, setup repo, API key | Repo siap |
| 2 | Pengumpulan data & anotasi (min. 3K) | `data/raw/` |
| 3 | EDA, preprocessing, baseline BiGRU/LSTM | Metrik baseline |
| 4 | Fine-tune IndoBERT, perbandingan model | Model terbaik |
| 5 | Deploy HF + skeleton ekstensi | Space + extension awal |
| 6 | Integrasi inferensi, UI hide/highlight | Prototype end-to-end |
| 7 | Testing & analisis hasil | Evaluasi lengkap |
| 8 | Laporan PDF & latihan presentasi | Semua deliverables |

---

## 13. Pembagian Tugas Kelompok (template)

| Anggota | Tanggung jawab |
|---------|----------------|
| Anggota 1 | Dataset: fetch API, labeling, EDA |
| Anggota 2 | Model baseline LSTM/GRU + evaluasi |
| Anggota 3 | Fine-tune IndoBERT + deploy Hugging Face |
| Anggota 4 | Ekstensi browser + integrasi inferensi |

---

## 14. Risiko dan Mitigasi

| Risiko | Mitigasi |
|--------|----------|
| Dataset judol kurang | Active collection + class weight / oversampling |
| Labeling lambat | Anotasi paralel sejak minggu 2 |
| Perubahan DOM YouTube | Selector cadangan + uji berkala |
| Model lambat di browser | Gunakan HF API atau model ringan |
| Kuota API habis | Cache hasil fetch, patuh ToS |

---

## 15. Poin Presentasi

1. Relevansi masalah di domain sosial media
2. Alur NLP: raw text → preprocess → tokenize → embedding → model → prediksi
3. Alasan memilih LSTM/GRU sebagai baseline
4. Alasan memilih Transformer/IndoBERT
5. Cara ekstensi membaca komentar di YouTube
6. Hasil evaluasi dan contoh error model
7. Demo live: hide / highlight komentar judol

---

## 16. Kesimpulan Rencana

Proyek ini memenuhi ketentuan final project NLP:
- **Domain:** Sosial Media (YouTube)
- **Task:** Text Classification (judol / bukan judol)
- **Teknologi:** BiGRU/LSTM + IndoBERT (HuggingFace)
- **Dataset:** ≥ 10.000 komentar berbahasa Indonesia
- **Aplikasi:** Ekstensi browser hide + highlight
- **Deploy:** Hugging Face Spaces + demo ekstensi

**Langkah berikutnya:** setup struktur folder repository dan mulai pengumpulan dataset.
