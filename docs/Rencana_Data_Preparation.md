# Rencana Persiapan Data (Data Preparation)

Dokumen ini berisi rencana dan kode untuk menggabungkan serta membersihkan dataset mentah (`datasetraw1.csv` dan `datasetraw2.csv`) sebelum masuk ke tahap *Feature Engineering*.

## 📋 Tahapan Persiapan

1.  **Standardisasi Kolom**: Menyamakan nama kolom dari kedua dataset agar bisa digabungkan.
2.  **Penggabungan (Merging)**: Menggabungkan dataset menjadi satu *dataframe* utama.
3.  **Penanganan Nilai Kosong (Handling Missing Values)**: Menghapus baris yang tidak memiliki teks komentar atau label.
4.  **Penanganan Duplikat**: Menghapus komentar yang berulang (spam seringkali identik).
5.  **Pembersihan Dasar**:
    *   Konversi ke huruf kecil (*lowercasing*).
    *   Penghapusan karakter non-alfanumerik dasar (opsional di tahap ini, tapi disarankan).
6.  **Validasi Label**: Memastikan label hanya berisi `0` (bukan judol) dan `1` (judol).
7.  **Penyimpanan**: Menyimpan hasil ke `Dataset/dataset_prepared.csv`.

---

## 💻 Kode Implementasi (Python)

Gunakan kode berikut di dalam notebook atau script Python:

```python
import pandas as pd
import re

# 1. Load Datasets
df1 = pd.read_csv('Dataset/datasetraw1.csv')
df2 = pd.read_csv('Dataset/datasetraw2.csv')

# 2. Standardisasi Kolom
# df1: komentar -> text, label -> label
# df2: message -> text, label -> label
df1_std = df1[['komentar', 'label']].rename(columns={'komentar': 'text', 'label': 'label'})
df2_std = df2[['message', 'label']].rename(columns={'message': 'text', 'label': 'label'})

# 3. Merging
df_combined = pd.concat([df1_std, df2_std], ignore_index=True)
print(f"Total data awal: {len(df_combined)}")

# 4. Handling Missing Values
df_combined = df_combined.dropna(subset=['text', 'label'])

# 5. Penanganan Duplikat
# Penting karena komentar judol seringkali berupa spam yang sama persis
df_combined = df_combined.drop_duplicates(subset=['text'])
print(f"Total data setelah hapus duplikat: {len(df_combined)}")

# 6. Pembersihan Dasar (Initial Cleaning)
def basic_clean(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_combined['text'] = df_combined['text'].apply(basic_clean)

# 7. Validasi Label
# Memastikan label adalah integer 0 atau 1
df_combined['label'] = df_combined['label'].astype(int)
df_combined = df_combined[df_combined['label'].isin([0, 1])]

# 8. Cek Distribusi Label
print("\nDistribusi Label:")
print(df_combined['label'].value_counts())

# 9. Simpan Dataset
output_path = 'Dataset/dataset_prepared.csv'
df_combined.to_csv(output_path, index=False)
print(f"\nDataset siap disimpan di: {output_path}")
```

## 🛠️ Langkah Selanjutnya
Setelah file `dataset_prepared.csv` tersedia, langkah berikutnya adalah:
- **Advanced Preprocessing**: Normalisasi slang (*slang mapping*), penghapusan stopword, dan *stemming/lemmatization*.
- **Feature Engineering**: Tokenisasi (IndoBERT tokenizer atau Word2Vec/TF-IDF untuk baseline).
