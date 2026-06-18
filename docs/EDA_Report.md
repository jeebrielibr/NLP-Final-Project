# Laporan EDA (Exploratory Data Analysis)

Laporan ini merangkum hasil analisis data pada komentar YouTube untuk deteksi judi online (*judol*). Analisis ini bertujuan untuk memahami karakteristik data sebelum masuk ke tahap pelatihan model.

---

## 1. Analisis Distribusi Label
Kita melakukan pemeriksaan keseimbangan kelas untuk mengetahui apakah dataset *imbalanced*.

**Hasil:**
- Label 0 (Normal): 88.4%
- Label 1 (Judol): 11.6%

**Insight:** Dataset mengalami ketidakseimbangan yang signifikan. Teknik mitigasi seperti *class weights* atau *oversampling* pada kelas `judol` akan diperlukan saat pelatihan model.

```python
# Cek distribusi
df['label'].value_counts(normalize=True)
```

---

## 2. Analisis Panjang Teks
Kita membandingkan distribusi jumlah kata antara komentar `judol` dan `normal`.

**Temuan:**
- Komentar **`judol`** lebih konsisten (rata-rata 10 kata, standar deviasi 5). Ini mendukung hipotesa bahwa pesan promosi judol seringkali berupa pesan *template*.
- Komentar **`normal`** sangat bervariasi (standar deviasi 17), dengan beberapa komentar sangat panjang.

```python
df['word_count'] = df['text'].str.split().str.len()
df.groupby('label')['word_count'].describe()
```

---

## 3. Analisis Kata Kunci (Text Mining)
Kita membandingkan kata-kata yang paling sering muncul untuk melihat apakah ada pola unik pada komentar `judol`.

**Temuan:**
- **Komentar Judol**: Didominasi oleh nama situs (contoh: `alexis17`, `sgi88`, `pulauwin`) dan kata-kata ajakan promosi (`bikin`, `main`). Emoji yang dikonversi (seperti `red_heart`, `star`, `fire`) juga menjadi fitur pembeda yang sangat kuat.
- **Komentar Normal**: Didominasi oleh bahasa percakapan sehari-hari (`bisa`, `sudah`, `iya`) dan ekspresi emosi (`face_with_tears_of_joy`).

```python
from collections import Counter

def get_top_n_words(corpus, n=20):
    words = ' '.join(corpus).split()
    return Counter(words).most_common(n)

# Bandingkan judol_words dan normal_words
```

---

## Kesimpulan & Rekomendasi
1.  **Preprocessing Valid**: Tokenisasi emoji menjadi teks (`demojize`) telah berhasil dan emoji-emoji tersebut (seperti `star`, `fire`) muncul sebagai indikator kuat untuk kelas `judol`.
2.  **Mitigasi Imbalance**: Karena label `judol` hanya ~11%, saat pelatihan model kita wajib menggunakan **Weighted Cross-Entropy** atau **SMOTE** untuk mencegah model bias ke kelas `normal`.
3.  **Feature Extraction**: Tokenizer yang kita gunakan harus tetap mempertahankan token khusus seperti `[URL]` dan token hasil konversi emoji karena terbukti memiliki nilai informatif yang tinggi.
