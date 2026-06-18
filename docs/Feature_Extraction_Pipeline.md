# Rencana & Dokumentasi: Feature Extraction Pipeline

Dokumen ini merinci langkah-langkah dalam proses ekstraksi fitur yang digunakan untuk mengubah teks komentar YouTube yang telah dibersihkan menjadi format numerik yang siap dilatih.

---

## 1. Analisis Distribusi Panjang Teks (EDA)
Langkah pertama adalah menentukan `MAX_LENGTH` yang optimal untuk proses padding/truncation agar tidak membuang informasi penting namun tetap efisien dalam penggunaan memori.

**Hasil Statistik:**
- Rata-rata kata: ~11 kata.
- 95th Percentile: 31 kata.
- 99th Percentile: 66 kata.
- **Rekomendasi `MAX_LENGTH`**: **100** (cukup untuk mencakup >99% data tanpa *padding* berlebih).

---

## 2. Split Dataset
Data dibagi menjadi tiga bagian dengan teknik *stratified sampling* untuk menjaga proporsi kelas `judol` (0/1).
*   **Train**: 70%
*   **Validation**: 15%
*   **Test**: 15%

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df['label'])
```

---

## 3. Tokenisasi Word-Level (Untuk BiGRU/LSTM)
Untuk model baseline BiGRU, kita memetakan setiap kata menjadi ID unik. Karena keterbatasan lingkungan, kita menggunakan *tokenizer* kustom `JudolTokenizer`.

```python
from collections import Counter
import numpy as np

class JudolTokenizer:
    def __init__(self, num_words=20000, oov_token='<OOV>'):
        self.word_index = {oov_token: 1}
        # ... (metode fit_on_texts dan texts_to_sequences)
```
*   **`num_words`**: 20.000 kata paling sering muncul.
*   **`OOV`**: Penanganan kata baru (*Out-Of-Vocabulary*).

---

## 4. Tokenisasi Subword (Untuk IndoBERT)
Untuk model Transformer, kita menggunakan *tokenizer* bawaan dari `indobenchmark/indobert-base-p2` agar kompatibel dengan arsitektur model.

```python
from transformers import AutoTokenizer

model_name = "indobenchmark/indobert-base-p2"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Konversi teks ke input_ids dan attention_mask
def prepare_indobert(texts):
    return bert_tokenizer(
        list(texts),
        max_length=100,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
```

---

## 5. Penyimpanan Asset
Hasil tokenisasi (terutama tokenizer untuk BiGRU) harus disimpan agar konsisten saat digunakan di *Browser Extension* nanti.

```python
import pickle
# Simpan tokenizer
with open('Dataset/tokenizer_bigru.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```
