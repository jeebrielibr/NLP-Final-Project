# Strategi Advanced Preprocessing: Deteksi Komentar Judol

Komentar judi online (judol) di YouTube memiliki karakteristik unik: penggunaan font unik (bold/italic unicode), emoji berlebihan, link yang disamarkan, dan istilah slang khusus. Strategi ini dirancang untuk menormalkan teks tersebut agar model NLP dapat menangkap fitur semantik dengan lebih baik.

---

## Tahap 1: Emoji Handling (Demojize)
**Masalah:** Emoji harus diproses paling awal. Jika kita melakukan normalisasi Unicode atau pembersihan ASCII di awal, emoji akan terhapus karena merupakan karakter non-ASCII.  
**Solusi:** Mengonversi emoji menjadi deskripsi teks menggunakan library `emoji` sebelum tahap pembersihan lainnya.

```python
import emoji

def convert_emoji(text):
    # Mengubah 🎰 menjadi :slot_machine: (ASCII safe)
    return emoji.demojize(text, delimiters=(" ", " "))
```

---

## Tahap 2: Unicode Normalization
**Masalah:** Spammer sering menggunakan karakter Unicode (misal: `𝙎𝙇𝙊𝙏`) untuk menghindari filter.  
**Solusi:** Setelah emoji aman dalam bentuk teks, kita bisa mengonversi font "fancy" ke bentuk dasar ASCII.

```python
import unicodedata

def normalize_unicode(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
```

---

## Tahap 4: Symbol Handling & Cleaning
**Masalah:** Simbol dekoratif (░, █) tidak memiliki arti semantik.  
**Solusi:** Menghapus simbol non-alfanumerik tetapi tetap mempertahankan token `[URL]` dan hasil demojize emoji (seperti `:slot_machine:`).

```python
def clean_symbols(text):
    # Menghapus simbol kecuali huruf, angka, spasi, [URL], dan underscore/colon (untuk emoji)
    text = re.sub(r'[^a-zA-Z0-9\s\[\]\_:]', ' ', text)
    return text
```

---

## Tahap 4: Karakter Berulang (Elongated Words)
**Masalah:** Penggunaan huruf berulang (misal: `gacorrrrr`, `menanggggg`).  
**Solusi:** Mengompres karakter yang berulang lebih dari 2 kali menjadi maksimal 2 karakter.

```python
def reduce_repeated_chars(text):
    # Contoh: "gacorrrrr" -> "gacorr"
    return re.sub(r'(.)\1{2,}', r'\1\1', text)
```

---

## Tahap 5: Normalisasi Teks dengan saka-nlp
**Masalah:** Komentar YouTube Indonesia penuh dengan slang (alay), singkatan, dan bahasa tidak baku.  
**Solusi:** Menggunakan library `saka-nlp` yang dirancang khusus untuk normalisasi teks bahasa Indonesia modern secara akurat.

```python
import saka

def normalize_with_saka(text):
    # saka.normalize menangani slang, singkatan, dan standarisasi kata
    return saka.normalize(text)
```

---

## Implementasi Pipeline Lengkap (Updated with saka-nlp)

Berikut adalah fungsi gabungan yang akan diterapkan pada `dataset_prepared.csv`:

```python
import saka
import unicodedata
import re
import emoji

def final_preprocessing_pipeline(text):
    if not isinstance(text, str): return ""
    
    # 1. Emoji to Text (Demojize) - Harus paling awal agar tidak terhapus proses ASCII
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # 2. Unicode Normalization (Bold/Italic -> Normal)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # 3. URL Masking
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
    text = re.sub(r'bit\s?[\.\,]\s?ly/\w+', '[URL]', text)
    text = re.sub(r'\w+\s?\[dot\]\s?\w+', '[URL]', text)
    
    # 4. Lowercase
    text = text.lower()
    
    # 5. Symbol & Cleaning (Keep alphanumeric, [URL], and underscore/colon from demojize)
    text = re.sub(r'[^a-zA-Z0-9\s\[\]\_:]', ' ', text)
    
    # 6. Normalize with saka-nlp (Indonesian slang)
    text = saka.normalize(text)
    
    # 7. Reduce repeated chars (e.g., menangggg -> menang)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---
**Target Output:** `Dataset/dataset_clean_final.csv`
- Kolom: `text` (hasil cleaning), `label` (0/1).
- Distribusi Data: Mempertahankan keseimbangan kelas atau melakukan sampling jika diperlukan.
