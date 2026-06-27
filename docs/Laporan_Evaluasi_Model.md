# Laporan Hasil Evaluasi Model

Laporan ini merangkum hasil evaluasi dari ketiga model klasifikasi yang dilatih untuk mendeteksi komentar judi online (*judol*) pada YouTube. Ketiga model dievaluasi pada dataset `dataset_clean_final.csv` yang berisi 70.379 baris komentar berbahasa Indonesia.

---

## 1. Metodologi Evaluasi

### 1.1 Dataset

| Aspek | Detail |
|-------|--------|
| **Nama file** | `dataset_clean_final.csv` |
| **Total baris** | 70.379 |
| **Label 0 (Bukan Judol)** | 62.202 (88,4%) |
| **Label 1 (Judol)** | 8.177 (11,6%) |
| **Kolom** | `text`, `label` |

Dataset mengalami ketidakseimbangan kelas yang signifikan (rasio ~7,6:1). Seluruh model menggunakan `stratify=y` pada saat splitting untuk menjaga rasio kelas pada data latih dan uji.

### 1.2 Strategi Splitting

| Model | Rasio Split | Random State | Keterangan |
|-------|-------------|--------------|------------|
| **BiGRU** | 80/20 | 42 | Train 56.303 · Test 14.076 |
| **LSTM** | 70/15/15 | 42 | Train 49.265 · Val 7.038 · Test 14.076 |
| **IndoBERT-focal** | 80/20 | 42 | Train 56.303 · Val/Test 14.076 |

> **Catatan:** BiGRU dan IndoBERT-focal menggunakan split yang identik (80/20, `random_state=42`) sehingga perbandingan antara keduanya bersifat adil. LSTM menggunakan rasio 70/15/15 dengan validation set terpisah, namun ukuran test set sama (14.076 baris).

### 1.3 Metrik Evaluasi

Metrik yang digunakan: **Accuracy**, **Precision**, **Recall**, **F1-Score** (per kelas dan makro/weighted average), serta **Confusion Matrix**. LSTM tambahan melaporkan **ROC-AUC Score**.

---

## 2. Hasil Evaluasi Model BiGRU

### 2.1 Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| **Framework** | PyTorch |
| **Vocab Size** | 23.154 kata |
| **Max Length** | 50 token |
| **Embedding Dim** | 128 |
| **Hidden Dim** | 64 |
| **Num Layers** | 2 (Bidirectional) |
| **Dropout** | 0,3 |
| **Loss Function** | BCEWithLogitsLoss |
| **Optimizer** | Adam (lr=0,001) |
| **Batch Size** | 64 |
| **Epochs** | 5 |
| **Device** | GPU (NVIDIA RTX 3050 6GB) |

### 2.2 Progress Training

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1 | 0,1509 | 95,00% | 0,0549 | 98,41% |
| 2 | 0,0587 | 98,23% | 0,0381 | 98,98% |
| 3 | 0,0405 | 98,87% | 0,0378 | 99,02% |
| 4 | 0,0314 | 99,13% | 0,0321 | 99,18% |
| 5 | 0,0254 | 99,28% | 0,0314 | 99,23% |

Model konvergen dengan cepat — akurasi test sudah melebihi 98% pada epoch pertama dan terus membaik hingga epoch ke-5. Tidak terdapat tanda *overfitting* yang signifikan (selisih Train Acc dan Test Acc < 0,1%).

### 2.3 Classification Report (Test Set)

```
              precision    recall  f1-score   support

 Bukan Judol       0.99      1.00      1.00     12441
       Judol       0.97      0.96      0.97      1635

    accuracy                           0.99     14076
   macro avg       0.98      0.98      0.98     14076
weighted avg       0.99      0.99      0.99     14076
```

### 2.4 Confusion Matrix (Derived)

Berdasarkan metrik di atas, nilai *confusion matrix* diturunkan sebagai berikut:

| | Prediksi: Bukan Judol | Prediksi: Judol |
|---|---|---|
| **Aktual: Bukan Judol** | 12.392 (TN) | 49 (FP) |
| **Aktual: Judol** | 65 (FN) | 1.570 (TP) |

### 2.5 Pengujian Inference

| Teks Komentar | Prediksi | Confidence |
|---------------|----------|------------|
| "Jangan lupa klaim bonus deposit pertama di situs zeus gampang menang hari ini!" | TERDETEKSI JUDOL | 99,13% |

---

## 3. Hasil Evaluasi Model LSTM

### 3.1 Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| **Framework** | TensorFlow / Keras |
| **Max Vocab** | 15.000 kata |
| **Max Length** | 100 token |
| **Embedding Dim** | 128 |
| **LSTM Units** | 64 (layer 1) · 32 (layer 2) |
| **Architecture** | Bidirectional LSTM (2 layer) |
| **Loss Function** | Binary Crossentropy |
| **Optimizer** | Adam (lr=0,001) |
| **Batch Size** | 128 |
| **Max Epochs** | 20 (Early Stopping) |
| **Class Weights** | {0: 0,567, 1: 4,303} |
| **Callbacks** | EarlyStopping (patience=4, monitor=val_auc), ReduceLROnPlateau |
| **Device** | CPU (TensorFlow GPU tidak tersedia di Windows native) |

### 3.2 Progress Training

Training dihentikan pada epoch ke-6 melalui *Early Stopping*. Bobot model dikembalikan ke epoch ke-2 (val_auc terbaik).

| Epoch | Train Acc | Train AUC | Val Acc | Val AUC | Val Loss | LR |
|-------|-----------|-----------|---------|---------|----------|-----|
| 1 | 93,61% | 0,9742 | 98,28% | 0,9916 | 0,0713 | 0,0010 |
| 2 ⭐ | 98,85% | 0,9965 | 98,61% | 0,9922 | 0,0578 | 0,0010 |
| 3 | 99,25% | 0,9987 | 98,51% | 0,9911 | 0,0647 | 0,0010 |
| 4 | 99,42% | 0,9994 | 98,01% | 0,9915 | 0,0705 | 0,0005 |
| 5 | 99,66% | 0,9998 | 98,64% | 0,9853 | 0,0706 | 0,0005 |
| 6 | 99,66% | 0,9997 | 98,86% | 0,9859 | 0,0704 | 0,00025 |

⭐ = Bobot model yang digunakan (best epoch)

Terdapat tanda *overfitting* mulai epoch ke-3: Train Acc terus naik (99,25% → 99,66%) sementara Val AUC menurun (0,9911 → 0,9859). *Early Stopping* bekerja sesuai harapan.

### 3.3 Classification Report (Test Set)

```
                precision    recall  f1-score   support

Bukan Judi (0)       1.00      0.99      0.99     12441
      Judi (1)       0.93      0.97      0.95      1635

      accuracy                           0.99     14076
     macro avg       0.96      0.98      0.97     14076
  weighted avg       0.99      0.99      0.99     14076
```

**ROC-AUC Score: 0,9953**

### 3.4 Confusion Matrix (Derived)

| | Prediksi: Bukan Judol | Prediksi: Judol |
|---|---|---|
| **Aktual: Bukan Judol** | 12.322 (TN) | 119 (FP) |
| **Aktual: Judol** | 49 (FN) | 1.586 (TP) |

### 3.5 Pengujian Inference

| Teks Komentar | Prediksi | Probabilitas |
|---------------|----------|--------------|
| "Daftar sekarang dan dapatkan bonus deposit 100% tanpa syarat!" | JUDI | 99,97% |
| "Makasih resepnya kak, enak banget dicoba di rumah" | BUKAN JUDI | 28,16% |
| "Link alternatif slot88 gacor hari ini, winrate tinggi!" | JUDI | 96,02% |
| "Semoga cepet sembuh ya, get well soon!" | BUKAN JUDI | 0,12% |
| "Bergabung sekarang, jackpot menanti kamu!" | JUDI | 99,90% |
| "Wah pemandangannya bagus banget, dimana nih lokasiknya?" | BUKAN JUDI | 9,90% |

---

## 4. Hasil Evaluasi Model IndoBERT + Focal Loss

### 4.1 Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| **Framework** | HuggingFace Transformers |
| **Pre-trained Model** | `indobenchmark/indobert-base-p1` |
| **Num Labels** | 2 |
| **Max Length** | 128 token |
| **Loss Function** | Focal Loss (custom WeightedFocalTrainer) |
| **Focal Loss Alpha** | [1,0 · 7,61] (rasio 62.227:8.177) |
| **Focal Loss Gamma** | 2,0 |
| **Optimizer** | AdamW (fused, lr=2e-5) |
| **Batch Size** | 32 (train) · 64 (eval) |
| **Epochs** | 5 |
| **Weight Decay** | 0,01 |
| **FP16** | Enabled |
| **Callbacks** | EarlyStoppingCallback (patience=2) |
| **Device** | GPU (NVIDIA RTX 3050 6GB) |
| **Training Runtime** | ~3.110 detik (±52 menit) |

### 4.2 Focal Loss — Rasional

Focal Loss diterapkan untuk mengatasi ketidakseimbangan kelas. Berbeda dari *Cross-Entropy* standar, Focal Loss memodifikasi loss dengan faktor `(1 - p_t)^γ` yang:

- **Mengecilkan loss** dari sampel yang mudah diklasifikasi (kelas mayoritas / komentar normal yang jelas bukan judol)
- **Membesarkan loss** dari sampel yang sulit (slang judol seperti "wd", "gacor", nama situs judi tersamar)

Bobot `alpha` memberikan penalti lebih besar untuk kesalahan pada kelas minoritas (judol), dengan rasio 7,61:1 sesuai proporsi dataset.

### 4.3 Training Loss

| Metrik | Nilai |
|--------|-------|
| **Training Loss** | 0,0195 |
| **Global Steps** | 8.800 |
| **Train Samples/Sec** | 90,51 |
| **Total FLOPs** | 1,85 × 10¹⁶ |

Training loss sangat rendah (0,0195), namun seperti yang akan ditunjukkan pada evaluasi test set, ini **tidak mencerminkan performa generalisasi model**.

### 4.4 Classification Report (Test Set)

```
--- TEST DATA EVALUATION REPORT (IndoBERT + Focal Loss) ---

              precision    recall  f1-score   support

 Bukan Judol       0.91      0.50      0.65     12441
       Judol       0.14      0.62      0.23      1635

    accuracy                           0.52     14076
   macro avg       0.53      0.56      0.44     14076
weighted avg       0.82      0.52      0.60     14076
```

### 4.5 Confusion Matrix

| | Prediksi: Bukan Judol | Prediksi: Judol |
|---|---|---|
| **Aktual: Bukan Judol** | 6.272 (TN) | 6.169 (FP) |
| **Aktual: Judol** | 614 (FN) | 1.021 (TP) |

### 4.6 Error Analysis

#### False Positive (6.169 kasus) — Komentar aman yang salah dikira judol

| Confidence | Teks Komentar |
|------------|---------------|
| 0,6563 | "saya mempunyai android nokia 6 nokia 5 3 dan sekarang memakai nokia 5 4 kualitas sungguh sangat luar" |
| 0,6541 | "review advan macha bang dari konten bang david" |
| 0,6504 | "korupsi lewat prosedur yaah muncul kan isu terorisme anggaran langsung cair ujung ujung nya di korup" |
| 0,6471 | "hahaha akhirnya sdh di fase ini dia parah men" |
| 0,6462 | "bang review smartwacht murah harga di bawah 100 yang bisa gps sama banyak olahraga" |

Model terlalu sensitif terhadap kata-kata seperti "review", "uang", "anggaran", "fase" yang tidak berkaitan dengan judol.

#### False Negative (614 kasus) — Komentar judol yang lolos deteksi

| Confidence | Teks Komentar |
|------------|---------------|
| 0,6680 | "best lah mandalika77" |
| 0,6441 | "habis buka pulauwin mood langsung balik" |
| 0,6153 | "dora77 maxwin terus" |
| 0,6069 | "saya pikir hype pulauwin ramai" |
| 0,5882 | "gak ada yang ngalahin probet855 beneran cuan tiap hari fire" |

Model gagal mengenali nama situs judi tersamar seperti `mandalika77`, `pulauwin`, `dora77`, `probet855`, `sgi88`, `alexis17` — yang justru merupakan pola judol paling umum di YouTube.

### 4.7 Pengujian Inference

| Teks Komentar | Prediksi | Confidence | Status |
|---------------|----------|------------|--------|
| "Wah videonya sangat edukatif, terima kasih bang!" | BUKAN_JUDOL | 98,66% | ✅ Benar |
| "Bongkar rahasia wd terus bosku, cek link di bio sekarang depo 10k jadi 100k" | JUDOL | 97,99% | ✅ Benar |
| "Gacor banget bang mainnya, tutor dong" | BUKAN_JUDOL | 96,88% | ❌ Salah (seharusnya Judol) |

---

## 5. Perbandingan Model

### 5.1 Tabel Perbandingan Metrik

| Metrik | BiGRU | LSTM | IndoBERT-focal |
|--------|-------|------|----------------|
| **Accuracy** | 99,23% | 99% | 52% |
| **Judol Precision** | 0,97 | 0,93 | 0,14 |
| **Judol Recall** | 0,96 | 0,97 | 0,62 |
| **Judol F1-Score** | 0,97 | 0,95 | 0,23 |
| **Bukan Judol F1** | 1,00 | 0,99 | 0,65 |
| **Macro F1** | 0,98 | 0,97 | 0,44 |
| **Weighted F1** | 0,99 | 0,99 | 0,60 |
| **False Positive** | 49 | 119 | 6.169 |
| **False Negative** | 65 | 49 | 614 |
| **ROC-AUC** | — | 0,9953 | — |

### 5.2 Tabel Perbandingan Arsitektur

| Aspek | BiGRU | LSTM | IndoBERT-focal |
|-------|-------|------|----------------|
| **Jenis Model** | RNN (GRU) | RNN (LSTM) | Transformer (BERT) |
| **Framework** | PyTorch | TensorFlow/Keras | HuggingFace Transformers |
| **Pre-trained** | Tidak (from scratch) | Tidak (from scratch) | Ya (indobert-base-p1) |
| **Tokenisasi** | Word-level | Word-level | Subword (WordPiece) |
| **Vocab Size** | 23.154 | 15.000 | 32K (bawaan BERT) |
| **Max Length** | 50 | 100 | 128 |
| **Penanganan Imbalance** | Tidak ada | Class Weights | Focal Loss + Alpha |
| **Early Stopping** | Tidak | Ya (patience=4) | Ya (patience=2) |
| **Epochs Aktual** | 5 | 6 (stop) | 5 |
| **Device** | GPU | CPU | GPU |

### 5.3 Analisis Perbandingan

#### BiGRU vs LSTM

Kedua model RNN mencapai performa yang sangat tinggi (99% accuracy) dengan metrik yang saling berdekatan. Perbedaan utama:

- **BiGRU unggul dalam precision Judol** (0,97 vs 0,93) — lebih sedikit *false positive*, cocok untuk mode *Highlight* di ekstensi.
- **LSTM unggul dalam recall Judol** (0,97 vs 0,96) — lebih sedikit *false negative*, cocok untuk mode *Hide* di ekstensi.
- LSTM menggunakan *class weights* yang memberikan recall lebih tinggi pada kelas minoritas, dengan trade-off precision yang sedikit lebih rendah.
- LSTM melaporkan ROC-AUC 0,9953 yang menunjukkan kemampuan diskriminasi yang sangat baik antara kelas judol dan bukan judol.

#### IndoBERT-focal: Kegagalan Generalisasi

IndoBERT-focal menunjukkan anomali yang menarik dan penting secara akademis:

1. **Training loss sangat rendah (0,0195)** — model "belajar" data latih dengan sangat baik.
2. **Test accuracy hanya 52%** — model gagal melakukan generalisasi.
3. **6.169 False Positive** — model memprediksi hampir setengah dari komentar normal sebagai judol.
4. **Masih 614 False Negative** — model juga masih melewatkan judol yang menggunakan nama situs tersamar.

Kombinasi Focal Loss dengan `alpha=7,61` dan `gamma=2,0` terlalu agresif: model menjadi *oversensitive* terhadap pola kata tertentu (seperti "review", "uang") dan mengklasifikasikannya sebagai judol, sekaligus masih gagal mengenali nama situs judi tersamar yang tidak muncul di data latih.

**Kesimpulan:** Focal Loss dengan konfigurasi saat ini menyebabkan model overfit pada pola tertentu di data latih dan kehilangan kemampuan generalisasi. Penyesuaian `alpha` dan `gamma`, atau penggunaan *class weights* standar (seperti pada LSTM) mungkin menjadi alternatif yang lebih stabil.

---

## 6. Kesimpulan & Rekomendasi

### 6.1 Kesimpulan

1. **BiGRU adalah model terbaik** secara keseluruhan dengan accuracy 99,23%, Judol F1 0,97, dan hanya 49 *false positive*. Model ini juga paling ringan dan cepat untuk inferensi.
2. **LSTM menjadi alternatif kuat** dengan recall Judol tertinggi (0,97) dan ROC-AUC 0,9953. Pemilihan antara BiGRU dan LSTM tergantung prioritas: precision (BiGRU) vs recall (LSTM).
3. **IndoBERT-focal gagal melakukan generalisasi** meskipun training loss rendah. Model ini tidak layak digunakan untuk deployment pada konfigurasi saat ini.
4. **Model RNN (from scratch) mengalahkan Transformer (pre-trained)** dalam kasus ini — temuan yang menarik secara akademis, kemungkinan karena:
   - Dataset cleaning sudah sangat baik (slang normalization, emoji handling) sehingga model sederhana sudah cukup.
   - Focal Loss yang terlalu agresif pada IndoBERT justru kontra-produktif.
   - Vocabulary kustom BiGRU/LSTM yang spesifik untuk domain judol lebih efektif dibanding subword tokenization BERT yang umum.

### 6.2 Rekomendasi

1. **Untuk deployment saat ini:** Gunakan BiGRU sebagai model utama di Hugging Face Spaces. Model sudah tersimpan di `model/bigru/` dengan performa terbaik.
2. **Untuk IndoBERT:** Perlu eksperimen ulang dengan:
   - `alpha` yang lebih moderat (mis. 3,0–4,0 alih-alih 7,61)
   - `gamma` yang lebih rendah (mis. 1,0–1,5)
   - Atau ganti Focal Loss dengan *weighted Cross-Entropy* standar
3. **Untuk laporan akademis:** Narasi kegagalan IndoBERT-focal → solusi Focal Loss justru memperburuk → BiGRU sederhana unggul adalah bukti pemahaman NLP yang mendalam dan jujur secara ilmiah.
4. **Ekstensi browser:** Mode *Highlight* sebaiknya menggunakan model dengan precision tinggi (BiGRU), sedangkan mode *Hide* dapat mempertimbangkan model dengan recall tinggi (LSTM) — tergantung apakah pengguna lebih toleran terhadap *false positive* atau *false negative*.
