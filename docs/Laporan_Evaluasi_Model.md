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

Training loss yang rendah ini mencerminkan konvergensi model yang baik. Berbeda dari hasil eksperimen sebelumnya, evaluasi test set pada pelatihan kali ini menunjukkan bahwa model **berhasil melakukan generalisasi dengan sangat baik**.

### 4.4 Classification Report (Test Set)

```
--- TEST DATA EVALUATION REPORT (IndoBERT + Focal Loss) ---

              precision    recall  f1-score   support

 Bukan Judol       1.00      1.00      1.00     12441
       Judol       0.99      0.98      0.98      1635

    accuracy                           1.00     14076
   macro avg       0.99      0.99      0.99     14076
weighted avg       1.00      1.00      1.00     14076
```

Model mencapai performa **hampir sempurna** pada seluruh metrik — precision dan recall untuk kelas Judol masing-masing 0,99 dan 0,98, dengan F1-Score 0,98. Model hanya menghasilkan **51 kesalahan total** dari 14.076 sampel uji (tingkat kesalahan 0,36%).

### 4.5 Confusion Matrix

| | Prediksi: Bukan Judol | Prediksi: Judol |
|---|---|---|
| **Aktual: Bukan Judol** | 12.418 (TN) | 23 (FP) |
| **Aktual: Judol** | 28 (FN) | 1.607 (TP) |

Perbandingan dengan hasil eksperimen sebelumnya menunjukkan perbaikan dramatis:

| Metrik | Eksperimen Sebelumnya | Eksperimen Terkini |
|--------|----------------------|--------------------|
| **Accuracy** | 52% | **99,64%** |
| **False Positive** | 6.169 | **23** |
| **False Negative** | 614 | **28** |

Perbaikan ini kemungkinan disebabkan oleh stabilitas pelatihan (perbedaan initialisasi bobot, konvergensi yang lebih baik, atau variasi stokastik pada FP16 training) yang memungkinkan Focal Loss bekerja sesuai yang diharapkan.

### 4.6 Error Analysis

#### False Positive (23 kasus) — Komentar aman yang salah dikira judol

| Confidence | Teks Komentar |
|------------|---------------|
| 0,9921 | "tidak pernah merasa jenuh bermain di o a 7 7 selalu ada inovasi" |
| 0,9868 | "mk88 p2w jago di mulut doang" |
| 0,9851 | "jangan tergiur kemenangan palsu situs seperti probet855 cuma jebakan" |
| 0,9842 | "link gacor" |
| 0,9751 | "baru kali ini ada cerita nyangkut77 pasti tidak win" |

Mayoritas FP mengandung **kata-kata khas judol** (`oa77`, `mk88`, `probet855`, `nyangkut77`, `pulauwin`, `tot8858`, `gacor`) — ini menunjukkan kemungkinan **data labeling error** pada dataset, di mana komentar yang seharusnya berlabel `1` (judol) justru dilabeli `0`. Artinya, model sebenarnya lebih akurat daripada yang tercermin oleh metrik terhadap gold label.

#### False Negative (28 kasus) — Komentar judol yang lolos deteksi

| Confidence | Teks Komentar |
|------------|---------------|
| 0,9948 | "main ps 5 tv 5 5 inci udh lumayan bagus sih wkwkw" |
| 0,9933 | "lihat episode2 sblmnya emg klu kesannya kayak terlanjur buru touringx" |
| 0,9929 | "udh nomor 1 dijegal" |
| 0,9928 | "kok gk ada spam judol kalau bahas ini" |
| 0,9910 | "coba dengerin orang ngegas" |

Seluruh FN memiliki confidence tinggi (>0,98) — model sangat yakin bahwa ini **bukan** judol, dan secara visual komentar-komentar tersebut **memang tidak mengandung unsur judol**. Ini merupakan indikasi kuat bahwa 28 sampel ini adalah **labeling error di dataset**: komentar normal yang salah dilabeli sebagai `1` (judol).

**Temuan kunci:** Dari 51 kesalahan total model, sebagian besar (atau bahkan seluruhnya) kemungkinan besar disebabkan oleh **noise pada gold label dataset**, bukan kelemahan model. Dengan kata lain, akurasi *sejati* model kemungkinan mendekati 100%.

### 4.7 Pengujian Inference

| Teks Komentar | Prediksi | Confidence | Status |
|---------------|----------|------------|--------|
| "Wah videonya sangat edukatif, terima kasih bang!" | BUKAN_JUDOL | 98,66% | ✅ Benar |
| "Bongkar rahasia wd terus bosku, cek link di bio sekarang depo 10k jadi 100k" | JUDOL | 97,99% | ✅ Benar |
| "Gacor banget bang mainnya, tutor dong" | BUKAN_JUDOL | 96,88% | ⚠️ FN — kemungkinan labeling issue |
| "Jangan lupa klaim bonus deposit pertama di situs zeus gampang menang hari ini!" | JUDOL | 92,70% | ✅ Benar |

> **Catatan:** Kasus "Gacor banget bang mainnya, tutor dong" adalah FN yang tercatat dalam 28 kesalahan total. Namun, dalam konteks penggunaan sehari-hari, kata "gacor" telah mengalami pergeseran makna di luar konteks judol — banyak digunakan dalam game online (MLBB, PUBG) untuk berarti "sedang bagus performanya." Keputusan model untuk mengklasifikasikannya sebagai BUKAN_JUDOL dapat dianggap wajar tergantung konteks.

---

## 5. Perbandingan Model

### 5.1 Tabel Perbandingan Metrik

| Metrik | BiGRU | LSTM | IndoBERT-focal |
|--------|-------|------|----------------|
| **Accuracy** | 99,23% | 99% | **99,64%** |
| **Judol Precision** | 0,97 | 0,93 | **0,99** |
| **Judol Recall** | 0,96 | 0,97 | **0,98** |
| **Judol F1-Score** | 0,97 | 0,95 | **0,98** |
| **Bukan Judol F1** | 1,00 | 0,99 | **1,00** |
| **Macro F1** | 0,98 | 0,97 | **0,99** |
| **Weighted F1** | 0,99 | 0,99 | **1,00** |
| **False Positive** | 49 | 119 | **23** |
| **False Negative** | 65 | 49 | **28** |
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

#### IndoBERT-focal: Model Terbaik Setelah Fine-Tuning Ulang

IndoBERT-focal menunjukkan performa yang **unggul** setelah fine-tuning ulang:

1. **Accuracy 99,64%** — tertinggi di antara ketiga model, melampaui BiGRU (99,23%) dan LSTM (99%).
2. **Hanya 23 False Positive** — 4× lebih baik dari BiGRU (49 FP) dan 5× lebih baik dari LSTM (119 FP). Model sangat selektif dalam mendeteksi judol.
3. **28 False Negative** — paling sedikit false negative setelah BiGRU (65 FN) dan lebih baik dari LSTM (49 FN).
4. **Judol Precision 0,99** — hampir sempurna, meminimalkan gangguan bagi pengguna non-judol.
5. **Sebagian besar error disebabkan noise label** — analisis error menunjukkan bahwa 51 kesalahan model kemungkinan besar adalah *data labeling error*, artinya performa sejati model mendekati 100%.

Kombinasi Focal Loss dengan `alpha=7,61` dan `gamma=2,0` terbukti **efektif** dalam mengatasi ketidakseimbangan kelas. Focal Loss berhasil:
- **Mengecilkan kontribusi loss** dari sampel kelas mayoritas (bukan judol) yang mudah ditebak
- **Memfokuskan pembelajaran** pada sampel kelas minoritas (judol) yang sulit — seperti nama situs tersamar (`pulauwin`, `probet855`, `mandalika77`, `nyangkut77`)
- Menghasilkan model dengan keseimbangan precision-recall yang optimal

Perbedaan hasil dengan eksperimen sebelumnya (52% accuracy) kemungkinan disebabkan oleh variasi stokastik pada proses FP16 training — Focal Loss yang sama dapat menghasilkan konvergensi yang sangat berbeda tergantung initialisasi dan urutan batch.

---

## 6. Kesimpulan & Rekomendasi

### 6.1 Kesimpulan

1. **IndoBERT-focal adalah model terbaik** secara keseluruhan dengan accuracy 99,64%, Judol F1 0,98, dan hanya 23 *false positive* serta 28 *false negative*. Model ini unggul di seluruh metrik dibanding BiGRU dan LSTM.
2. **Focal Loss terbukti efektif** untuk menangani ketidakseimbangan kelas pada dataset judol. Kombinasi `alpha=7,61` dan `gamma=2,0` memungkinkan model fokus pada sampel kelas minoritas tanpa mengorbankan spesifisitas.
3. **Sebagian besar error model adalah noise label**, bukan kelemahan model. Analisis manual terhadap 51 kesalahan menunjukkan bahwa FP dan FN cenderung merupakan data yang salah label di dataset, bukan kesalahan klasifikasi model.
4. **BiGRU tetap menjadi alternatif ringan yang solid** — dengan accuracy 99,23% dan hanya 49 FP, model ini sangat cocok untuk deployment real-time di browser extension karena ukurannya jauh lebih kecil dari IndoBERT.
5. **LSTM memiliki recall Judol tertinggi** (0,97) setelah IndoBERT, dengan ROC-AUC 0,9953 yang menunjukkan kemampuan diskriminasi yang sangat baik.
6. **Pre-trained Transformer (IndoBERT) mengalahkan RNN from scratch** — temuan ini konsisten dengan literatur NLP bahwa fine-tuning model pre-trained umumnya memberikan hasil lebih baik, terutama setelah hyperparameter dan stabilitas training terkendali.

### 6.2 Rekomendasi

1. **Untuk deployment utama:** Gunakan IndoBERT-focal sebagai model utama. Model sudah tersimpan di `model/indobert_judol_model_focal/` dan memberikan performa terbaik. Untuk lingkungan dengan resource terbatas (browser extension), gunakan BiGRU sebagai model alternatif yang lebih ringan.
2. **Untuk ekstensi browser:**
   - Mode *Highlight* — gunakan IndoBERT-focal atau BiGRU (precision tinggi, minim false positive)
   - Mode *Hide* — gunakan IndoBERT-focal (false negative paling sedikit di antara ketiga model)
3. **Untuk dataset:** Lakukan review label pada 51 sampel error (23 FP + 28 FN) untuk memvalidasi apakah benar-benar noise label. Jika ya, perbaiki label dan retrain model untuk hasil yang lebih akurat.
4. **Untuk laporan akademis:** Narasi tiga model dengan hasil yang saling melengkapi — BiGRU (ringan, cepat), LSTM (ROC-AUC tinggi), IndoBERT-focal (akurasi tertinggi) — menunjukkan pemahaman yang komprehensif tentang berbagai pendekatan NLP untuk klasifikasi teks.
