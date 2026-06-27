# Rencana Migrasi Model Deployment: IndoBERT-focal → BiGRU

Dokumen ini merinci rencana penggantian model yang di-deploy pada Hugging Face Spaces dari IndoBERT + Focal Loss ke BiGRU. **Kode deployment belum diubah** — menunggu instruksi lanjutan.

---

## 1. Latar Belakang & Justifikasi

Berdasarkan [Laporan Evaluasi Model](Laporan_Evaluasi_Model.md), IndoBERT-focal yang saat ini di-deploy mengalami kegagalan generalisasi yang parah:

| Metrik | IndoBERT-focal (current) | BiGRU (target) |
|--------|:------------------------:|:---------------:|
| **Accuracy** | 52% | 99,23% |
| **Judol F1** | 0,23 | 0,97 |
| **False Positive** | 6.169 | 49 |
| **False Negative** | 614 | 65 |

Ekstensi browser saat ini memanggil API yang melayani model dengan **52% accuracy** — setara lemparan koin. Migrasi ke BiGRU akan meningkatkan kualitas deteksi ekstensi secara drastis.

**Keunggulan tambahan BiGRU:**
- Model jauh lebih kecil (~3M params vs ~110M params IndoBERT)
- Tidak memerlukan library `transformers` → Docker image lebih ringan
- Cold start lebih cepat di HF Spaces (tidak perlu download model pre-trained)
- Inferensi lebih cepat (word-level tokenization vs subword WordPiece)

---

## 2. Kondisi Deployment Saat Ini

### 2.1 File yang Terlibat

```
deployhf/
├── app.py              # FastAPI + HF pipeline (IndoBERT)
├── Dockerfile           # Python 3.10-slim, port 7860
└── requirements.txt     # fastapi, uvicorn, transformers, torch, pydantic
```

### 2.2 Cara Kerja API Saat Ini

```python
# app.py (current)
from transformers import pipeline
classifier = pipeline("text-classification", model="./", tokenizer="./")

@app.post("/predict")
def predict_comment(req: CommentRequest):
    result = classifier(req.text)[0]
    raw_label = result['label'].upper()
    label = "judol" if raw_label == "JUDOL" else "bukan_judol"
    return {"label": label, "score": float(result['score'])}
```

API menerima teks yang sudah dipreprocess oleh ekstensi (`preprocess.js`), lalu meneruskan langsung ke HuggingFace pipeline. Pipeline melakukan subword tokenization (WordPiece) dan inferensi IndoBERT.

### 2.3 API Contract (Harus Tetap Sama)

Ekstensi (`service-worker.js`) mengharapkan format respons berikut:

```
POST /predict
Request:  {"text": "komentar youtube"}
Response: {"label": "judol" | "bukan_judol", "score": 0.98}
```

Format ini **tidak boleh berubah** agar ekstensi tetap berfungsi tanpa modifikasi.

---

## 3. Kondisi Target (BiGRU)

### 3.1 Model Files

File model BiGRU tersimpan di `model/bigru/`:

| File | Isi |
|------|-----|
| `bigru_model.pt` | Model state_dict (weights) |
| `bigru_config.pt` | Config dict: `{vocab, MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, SLANG_JUDOL_MAP}` |

Nilai config:
- `VOCAB_SIZE` = 23.154
- `EMBEDDING_DIM` = 128
- `HIDDEN_DIM` = 64
- `MAX_LEN` = 50
- `output_dim` = 1
- `num_layers` = 2
- `dropout` = 0,3

### 3.2 Arsitektur Model (Harus Direplikasi di API)

```python
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        return self.fc(self.dropout(hidden_concat))
```

### 3.3 Preprocessing & Tokenization (Harus Direplikasi di API)

Fungsi `clean_text()` dan `text_to_sequence()` dari `GRU_model.ipynb`:

```python
SLANG_JUDOL_MAP = {
    'gacor': 'menang mudah',
    'wd': 'withdraw penarikan dana',
    'depo': 'deposit pengisian dana',
    'judol': 'judi online',
    's1ot': 'slot',
    'sl0t': 'slot',
    'jp': 'jackpot',
    'zeus': 'judi kakek zeus'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', 'link_website', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    normalized_words = [SLANG_JUDOL_MAP.get(word, word) for word in words]
    return " ".join(normalized_words)

def text_to_sequence(text, vocab, max_len):
    tokens = text.split()
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(sequence) < max_len:
        sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence
```

Inferensi:

```python
model.eval()
cleaned = clean_text(text)
seq = text_to_sequence(cleaned, vocab, MAX_LEN)
tensor_input = torch.tensor([seq], dtype=torch.long)
with torch.no_grad():
    prediction = torch.sigmoid(model(tensor_input)).item()
# prediction >= 0.5 → judol
```

---

## 4. Perubahan yang Diperlukan per File

### 4.1 `deployhf/app.py` — Rewrite Total

| Aspek | Saat Ini | Target |
|-------|----------|--------|
| Import | `from transformers import pipeline` | `import torch, torch.nn as nn, re` |
| Model loading | `pipeline(...)` satu baris | Load `BiGRUClassifier` class + `load_state_dict` + `bigru_config.pt` |
| Preprocessing | Tidak ada (didelegasikan ke HF tokenizer) | `clean_text()` + `text_to_sequence()` |
| Inference | `classifier(text)[0]` | `torch.sigmoid(model(tensor))` → threshold 0,5 |
| Response format | `{"label", "score"}` | Sama — tidak berubah |

### 4.2 `deployhf/requirements.txt` — Sederhana

| Dependensi | Status |
|------------|--------|
| `fastapi==0.111.0` | Tetap |
| `uvicorn==0.30.1` | Tetap |
| `torch==2.3.1` | Tetap |
| `pydantic==2.7.4` | Tetap |
| `transformers==4.41.2` | **Dihapus** — tidak lagi diperlukan |

Penghapusan `transformers` mengurangi ukuran Docker image signifikan (transformers + dependensinya ~500MB+).

### 4.3 `deployhf/Dockerfile` — Tidak Berubah

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 4.4 Model Files — Copy ke `deployhf/`

File berikut perlu disertakan di repo Hugging Face Space:

```
deployhf/
├── app.py
├── Dockerfile
├── requirements.txt
├── bigru_model.pt       ← copy dari model/bigru/
└── bigru_config.pt      ← copy dari model/bigru/
```

Atau set `model_path = "./model/bigru/"` jika struktur folder dipertahankan.

---

## 5. Analisis Preprocessing Chain

### 5.1 Alur Preprocessing Saat Ini

```
YouTube comment (raw)
    ↓
Extension preprocess.js     ← emoji, unicode, URL→[URL], slang, char reduction
    ↓
API (app.py)                ← diteruskan langsung ke HF pipeline
    ↓
HF tokenizer (WordPiece)    ← subword tokenization
    ↓
IndoBERT model
```

### 5.2 Alur Preprocessing Target

```
YouTube comment (raw)
    ↓
Extension preprocess.js     ← emoji, unicode, URL→[URL], slang, char reduction
    ↓
API (app.py)                ← clean_text() + text_to_sequence()
    ↓
BiGRU model
```

### 5.3 Mismatch yang Teridentifikasi

BiGRU dilatih pada `dataset_clean_final.csv` (output `AdvancedPreprocessing.ipynb`), lalu diterapkan `clean_text()` di notebook training. Ekstensi menggunakan `preprocess.js` yang mirip tapi tidak identik dengan `AdvancedPreprocessing.ipynb`. Berikut perbedaan yang memengaruhi:

| Aspek | Training (AdvancedPreprocessing + clean_text) | Ekstensi (preprocess.js) | Dampak |
|-------|-----------------------------------------------|--------------------------|--------|
| **URL token** | `link_website` | `[URL]` → `clean_text` strip brackets → `url` | `url` ≠ `link_website` → OOV → `<UNK>`. Dampak kecil: model melihat `<UNK>` untuk URL, masih bisa klasifikasi dari kata lain. |
| **Slang normalization** | saka-nlp + SLANG_JUDOL_MAP (8 entri, mapping berbeda) | SLANG_MAP JS (~150 entri, mapping berbeda) | Ekstensi sudah menormalisasi `wd`→`withdraw` sebelum API. Lalu `clean_text` mencari `wd` di SLANG_JUDOL_MAP — tidak ketemu karena sudah jadi `withdraw`. Hasil: `withdraw` masuk vocab (jika ada di training data) atau `<UNK>`. |
| **Symbol retention** | `clean_text` strip semua kecuali `[a-z0-9\s]` | `preprocess.js` retain `[`, `]`, `_`, `:` | Setelah `clean_text` di API: `[`, `]`, `_`, `:` di-strip. Konsisten dengan training. |

### 5.4 Penilaian Risiko Preprocessing

**Risiko: RENDAH** — ketiga mismatch di atas menghasilkan token `<UNK>` untuk sebagian kecil kata, namun model BiGRU terbukti robust dengan 99,23% accuracy. Vocab 23.154 kata mencakup mayoritas kata penting. Kata kunci judol seperti `depo`, `wd`, `gacor`, `slot`, `jackpot` ada di vocab.

**Mitigasi opsional (jika diperlukan):** Tambahkan mapping `[URL]` → `link_website` di awal `clean_text()` di API, sebelum regex strip simbol. Ini akan menyamakan token URL dengan training data.

---

## 6. Estimasi Ukuran & Performa

### 6.1 Ukuran Model

| Model | Ukuran File | Params |
|-------|-------------|--------|
| IndoBERT-focal | ~420 MB (model.safetensors) | ~110M |
| BiGRU | ~12 MB (bigru_model.pt + bigru_config.pt) | ~3M |

### 6.2 Docker Image

| Komponen | Saat Ini | Target |
|----------|----------|--------|
| Base image | python:3.10-slim (~150MB) | Sama |
| torch | ~800MB | Sama |
| transformers + deps | ~500MB | **Dihapus** |
| Model files | ~420MB | ~12MB |
| **Total estimasi** | **~1,9GB** | **~960MB** |

### 6.3 Cold Start & Inference

| Metrik | IndoBERT-focal | BiGRU |
|--------|----------------|-------|
| **Cold start** | 2–4 menit (load 110M params + tokenizer) | <30 detik (load 3M params) |
| **Inference per comment** | ~200–500ms (subword tokenization + transformer forward) | ~5–20ms (word split + GRU forward) |
| **Memory usage** | ~1–2GB | ~200–400MB |

---

## 7. Risiko & Mitigasi

| # | Risiko | Severity | Mitigasi |
|---|--------|----------|----------|
| 1 | **Preprocessing mismatch** antara ekstensi JS dan API Python | Rendah | Model robust (99% acc). Opsional: tambah mapping `[URL]`→`link_website` di `clean_text()` API |
| 2 | **OOV words** — kata tidak di vocab masuk `<UNK>` | Rendah | Vocab 23K kata mencakup kata kunci judol penting. `<UNK>` adalah token yang sudah dikenal model saat training |
| 3 | **Model architecture mismatch** — class definition di API tidak sama persis dengan training | Sedang | Copy paste persis dari `GRU_model.ipynb`. Verifikasi dengan inference test sebelum deploy |
| 4 | **Ekstensi tidak perlu diubah** — API contract sama | — | Verifikasi: response format `{label, score}` tetap sama. Service-worker.js tidak perlu diubah |
| 5 | **HF Space perlu re-deploy** dengan file model baru | Sedang | Push file `bigru_model.pt` + `bigru_config.pt` ke repo Space. Hapus file IndoBERT lama |

---

## 8. Langkah Implementasi (Menunggu Instruksi)

Berikut urutan langkah yang akan dijalankan setelah instruksi diberikan:

1. **Salin model files** — copy `model/bigru/bigru_model.pt` dan `model/bigru/bigru_config.pt` ke `deployhf/`
2. **Rewrite `deployhf/app.py`** — ganti HF pipeline dengan:
   - Definisi class `BiGRUClassifier`
   - Load `bigru_config.pt` (vocab, MAX_LEN, dll)
   - Load `bigru_model.pt` (state_dict)
   - Fungsi `clean_text()` + `text_to_sequence()`
   - Endpoint `/predict` dengan inferensi `torch.sigmoid(model(tensor))`
   - Threshold 0,5 → `judol` / `bukan_judol`
   - Response format tetap `{"label", "score"}`
3. **Update `deployhf/requirements.txt`** — hapus `transformers`
4. **Verifikasi lokal** — jalankan API secara lokal, test dengan contoh komentar:
   - "Jangan lupa klaim bonus deposit pertama di situs zeus gampang menang hari ini!" → `judol`
   - "Wah videonya sangat edukatif, terima kasih bang!" → `bukan_judol`
5. **Push ke Hugging Face Space** — deploy perubahan
6. **Test ekstensi** — verifikasi ekstensi tetap berfungsi tanpa perubahan kode

### 8.1 File yang Tidak Berubah

| File | Status |
|------|--------|
| `deployhf/Dockerfile` | Tidak berubah |
| `extension/**` | Tidak berubah — API contract sama |
| `model/bigru/*` | Tidak berubah — hanya di-copy ke `deployhf/` |

### 8.2 File yang Berubah

| File | Perubahan |
|------|-----------|
| `deployhf/app.py` | Rewrite total |
| `deployhf/requirements.txt` | Hapus `transformers` |
| `deployhf/bigru_model.pt` | Baru — copy dari `model/bigru/` |
| `deployhf/bigru_config.pt` | Baru — copy dari `model/bigru/` |

---

## 9. Struktur `app.py` Target (Pseudocode)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, torch.nn as nn, re, collections

# --- Model definition (copy from GRU_model.ipynb) ---
class BiGRUClassifier(nn.Module): ...

# --- Preprocessing (copy from GRU_model.ipynb) ---
SLANG_JUDOL_MAP = { ... }
def clean_text(text): ...
def text_to_sequence(text, vocab, max_len): ...

# --- Load model & config on startup ---
config = torch.load("bigru_config.pt")
vocab = config["vocab"]; MAX_LEN = config["MAX_LEN"]; ...
model = BiGRUClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, 1)
model.load_state_dict(torch.load("bigru_model.pt", map_location="cpu"))
model.eval()

# --- API endpoints (same contract as before) ---
@app.post("/predict")
def predict_comment(req: CommentRequest):
    cleaned = clean_text(req.text)
    seq = text_to_sequence(cleaned, vocab, MAX_LEN)
    tensor_input = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor_input)).item()
    label = "judol" if score >= 0.5 else "bukan_judol"
    return {"label": label, "score": score}
```
