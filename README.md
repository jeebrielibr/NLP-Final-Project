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

## Tujuan

| # | Tujuan | Indikator Keberhasilan |
|---|--------|------------------------|
| 1 | Mendeteksi komentar judol secara otomatis | Model klasifikasi berjalan end-to-end |
| 2 | Mengurangi paparan komentar judol di YouTube | Mode hide aktif di ekstensi |
| 3 | Membuktikan deteksi berhasil | Mode highlight menampilkan label/kelas |
| 4 | Memenuhi syarat mata kuliah NLP | Pipeline NLP jelas, evaluasi terukur, deploy tersedia |

---

## Fitur Utama

### Model NLP

- Klasifikasi biner: `judol` vs `bukan_judol`
- Baseline **BiGRU/LSTM** (wajib) dan fine-tune **IndoBERT/mBERT** (HuggingFace)
- Preprocessing teks Indonesia informal (slang, URL, emoji, typo)
- Evaluasi: accuracy, precision, recall, F1-score, confusion matrix

### Ekstensi Browser (Chrome/Edge)

| Fitur | Deskripsi |
|-------|-----------|
| **Hide** | Menyembunyikan komentar terdeteksi judol |
| **Highlight** | Menyorot komentar dengan border/warna + label |
| Toggle mode | Pengguna memilih hide atau highlight |
| Threshold confidence | Filter prediksi dengan skor minimum |
| Statistik sesi | Jumlah komentar terdeteksi per halaman video |
| Enable / disable | Master switch tanpa uninstall |

---

## Arsitektur Sistem

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

### Alur Runtime (saat pengguna membuka YouTube)

1. Ekstensi membaca komentar dari halaman YouTube (Content Script / DOM observer)
2. Teks komentar dikirim ke modul inferensi (lokal via ONNX/Web, atau API Hugging Face)
3. Model mengembalikan prediksi: `judol` atau `bukan_judol`
4. Ekstensi menerapkan aksi sesuai pengaturan: **Hide** atau **Highlight**

### Alur Offline (pelatihan model)

1. **Fetch data** — komentar YouTube via YouTube Data API v3
2. **Preprocessing** — cleaning, normalisasi slang, tokenisasi, handling URL/emoji
3. **Labeling** — anotasi manual: `judol` vs `bukan_judol`
4. **Training** — baseline BiGRU/LSTM + fine-tune IndoBERT/mBERT
5. **Evaluasi** — metrik klasifikasi + analisis error
6. **Export** — model ke format deploy (Hugging Face Inference API atau ONNX)

---

## Pipeline NLP

| Langkah | Keterangan |
|---------|------------|
| Lowercasing / normalisasi | Standarisasi teks Indonesia informal |
| URL dan link normalization | Pola judol sering mengandung link |
| Emoji handling | Pertahankan atau normalisasi sebagai fitur |
| Slang dan typo mapping | Kamus slang judol (mis. "wd", "depo", "slot gacor") |
| Tokenisasi | Word-level (LSTM) dan subword/BPE (Transformer) |
| Padding / truncation | Panjang maksimum token disesuaikan distribusi komentar |

### Model

| Model | Peran |
|-------|-------|
| **BiGRU / LSTM** | Baseline sequence model — ringan, mudah dijelaskan alur embedding → RNN → dense |
| **IndoBERT / mBERT** | Model utama performa tinggi — kuat untuk konteks bahasa Indonesia |

---

## Dataset

| Aspek | Detail |
|-------|--------|
| **Minimal** | 10.000 data berbahasa Indonesia |
| **Sumber** | YouTube Data API v3, anotasi manual kelompok |
| **Label `judol`** | Promosi judi online, link situs judi, ajakan deposit/main slot/casino |
| **Label `bukan_judol`** | Komentar normal, feedback video, diskusi umum |
| **Pembagian** | Train 70% · Validation 15% · Test 15% |

---

## Tech Stack

| Lapisan | Teknologi |
|---------|-----------|
| Training | Python 3.10+, PyTorch, HuggingFace Transformers |
| Baseline RNN | BiGRU/LSTM (PyTorch atau Keras) |
| Data & evaluasi | pandas, scikit-learn, matplotlib, seaborn |
| API dataset | YouTube Data API v3 |
| Deploy model | Hugging Face Spaces |
| Ekstensi | Chrome Extension Manifest V3 (JavaScript) |

---

## Struktur Repository

```
NLP Final Project/
├── Docs/
│   ├── Desc.md              # Ketentuan mata kuliah & anggota
│   └── Rencana_Proyek.md    # Rencana detail proyek
├── Dataset/                 # Dataset mentah & terproses
├── Notebook/                # EDA, training, evaluasi
├── src/                     # Kode sumber pipeline NLP
├── extension/               # Ekstensi browser Chrome/Edge
├── deployment/huggingface/  # Konfigurasi deploy Hugging Face
├── reports/                 # Laporan & artefak presentasi
├── requirements.txt
└── README.md
```

---

## Ketentuan Mata Kuliah

Proyek ini memenuhi ketentuan final project NLP:

- **Domain:** Sosial Media (YouTube)
- **Task:** Text Classification (`judol` / `bukan_judol`)
- **Teknologi:** BiGRU/LSTM (wajib) + IndoBERT/HuggingFace
- **Dataset:** ≥ 10.000 komentar berbahasa Indonesia
- **Deploy:** Minimal Hugging Face Spaces + demo ekstensi browser

Detail lengkap ketentuan, bobot penilaian, dan format laporan ada di [`Docs/Desc.md`](Docs/Desc.md).

---

## Deliverables

- [ ] Repository kode lengkap
- [ ] Model + demo Hugging Face
- [ ] Ekstensi browser (hide + highlight)
- [ ] Laporan PDF 5–10 halaman
- [ ] Slide presentasi

---

## Timeline (6–8 Minggu)

| Minggu | Kegiatan | Output |
|--------|----------|--------|
| 1 | Finalisasi rencana, setup repo, API key | Repo siap |
| 2 | Pengumpulan data & anotasi (min. 3K) | `Dataset/` |
| 3 | EDA, preprocessing, baseline BiGRU/LSTM | Metrik baseline |
| 4 | Fine-tune IndoBERT, perbandingan model | Model terbaik |
| 5 | Deploy HF + skeleton ekstensi | Space + extension awal |
| 6 | Integrasi inferensi, UI hide/highlight | Prototype end-to-end |
| 7 | Testing & analisis hasil | Evaluasi lengkap |
| 8 | Laporan PDF & latihan presentasi | Semua deliverables |

---

## Dokumentasi

- [Rencana Proyek](Docs/Rencana_Proyek.md) — arsitektur, pipeline, evaluasi, dan rencana implementasi
- [Ketentuan Mata Kuliah](Docs/Desc.md) — informasi kelompok, syarat, dan bobot penilaian

---

## Lisensi

Proyek akademik — Teknik Informatika, mata kuliah Natural Language Processing.
