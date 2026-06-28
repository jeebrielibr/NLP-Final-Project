# 🎨 Panduan Desain Ekstensi: Judol Remover (Pop-Art Cosmic Theme)

Dokumen ini mendokumentasikan spesifikasi desain antarmuka (UI/UX) untuk Ekstensi Browser **Judol Remover** versi **v1.0.0 (Kelompok Newton)**. Tema ini mengusung konsep **Artistic, Penuh Warna, Kreatif, dan Kaya Emoji** dengan basis visual *Dark-Mode Doodle Art*.

---

## 👁️ Ringkasan Konsep

| Komponen | Spesifikasi |
| :--- | :--- |
| **Gaya Utama** | *Doodle Pop-Art* dengan kontur garis hitam tegas (*bold outlines*) |
| **Nuansa Latar** | *Cyber Charcoal / Deep Space Dark* (Kontras tinggi terhadap warna neon) |
| **Elemen Grafis** | Ilustrasi kosmik (planet, roket, bintang), ekspresi wajah (*doodle faces*) |
| **Interaktivitas** | Kaya akan *emoji* kontekstual, efek *glow* neon, dan kartu dinamis |

---

## 🎨 Palet Warna (Color Palette)

Aplikasi menggunakan kombinasi warna latar belakang gelap yang tenang dengan aksen warna-warni pop-art yang mencolok (desaturasi terkontrol agar tidak merusak mata namun tetap kontras dan ceria).

* **Background Utama (Doodle Wallpaper):** `#0C1020` (Gelap kebiruan dengan pola coretan warna mentereng kuning `#FBD835`, pink `#F44090`, cyan `#21CBD3`).
* **Background Kartu (Card Containers):** `#151E33` dengan radius sudut (*border-radius*) `16px` dan border tipis `#26375A`.
* **Warna Aksen & Status:**
    * 🟢 **Active / Success (Hijau Neon):** `#63E660` — Digunakan untuk tombol aktif dan status API tersedia.
    * 🔵 **Total Checked (Biru Cerah):** `#3BA3F3` — Digunakan untuk background statistik total diperiksa.
    * 🔴 **Alert / Danger (Merah Coral):** `#EC4D3D` — Digunakan untuk panel statistik judi online yang terdeteksi.
    * 🟡 **Warning / Highlight (Kuning Emas):** `#F9D034` — Digunakan untuk tanda-tanda perhatian atau elemen dekoratif.

---

## 🧩 Komponen UI & Penggunaan Emoji

### 1. Header Utama
* **Logo & Nama:** Menggunakan ikon perisai mosaik warna-warni di samping teks **"Judol Remover"** (Putih, Bold, `22pt`).
* **Sub-Header:** Teks deskripsi kecil `"Deteksi komentar judi online di YouTube"` dengan warna abu-abu terang.
* **Tombol Utama (Toggle Switch):** * Label: `Aktifkan Ekstensi`
    * Ikon Sekitar: 🌍 (Bumi), 🚀 (Roket), 🪐 (Saturnus) untuk menghidupkan suasana kosmik.
    * Warna Switch: Hijau neon (`#63E660`) saat aktif dengan knob putih bulat.

### 2. Panel Mode Tampilan (Display Mode)
Pilihan interaktif menggunakan kontainer berbayang neon (*glow effect*):
* **Opsi 1: Highlight 🔍🎉**
    * Teks: *Tandai komentar judol dengan border + label*
    * Efek Visual: Border luar berwarna hijau pendar berpendar (*neon green glow*).
* **Opsi 2: Hide 🎭👻**
    * Teks: *Sembunyikan komentar judol sepenuhnya*
    * Emoji Tambahan: Topeng pesta (🎭) dan hantu lucu (👻) untuk melambangkan fitur menghilang/bersembunyi.

### 3. Sliders: Confidence Threshold 👶🤔🧠
Pengaturan tingkat keyakinan model kecerdasan buatan (NLP) disajikan dengan gradien warna horizontal (Pelangi dari Kuning ke Ungu-Merah) dari rentang `0.0` ke `1.0`:
* **0.0 (Sangat Sensitif):** Ditandai dengan emoji Bayi (👶) — Model sangat polos/sensitif menangkap segala hal.
* **0.5 (Seimbang):** Ditandai dengan emoji Berpikir (🤔) — Model mempertimbangkan dengan matang.
* **1.0 (Paling Pasti):** Ditandai dengan emoji Otak (🧠) — Hanya prediksi yang berbobot dan sangat pasti yang akan dieksekusi.

### 4. Status API
* **Komponen:** Kotak panjang berwarna hijau dengan tombol radio aktif.
* **Teks:** `API tersedia` dengan emoji jempol besar (👍) dan ikon rak server komputer (🖥️/🎛️).

### 5. Statistik Sesi (Session Stats)
Dua buah kartu berdampingan dengan ukuran simetris namun warna kontras untuk mempercepat pemindaian informasi visual:
1.  **Judol Terdeteksi (Kartu Merah 🔴):** Background `#EC4D3D` dilengkapi dengan kluster emoji wajah marah (😡😡😡) di sudut atas kartu.
2.  **Total Diperiksa (Kartu Biru 🔵):** Background `#3BA3F3` dilengkapi dengan sepasang emoji mata melirik (👀) di atas kartu, menggambarkan sistem yang terus mengawasi halaman.

---

## 🛠️ Panduan CSS & Styling Kode

Untuk mengimplementasikan tema ini ke dalam file HTML/CSS ekstensi (`popup.html` dan `popup.css`), gunakan referensi struktur gaya berikut:
```css
/* popup.css */
body {
    background-color: #0c1020;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #ffffff;
    margin: 0;
    padding: 15px;
    width: 360px; /* Ukuran standar popup ekstensi */
}

.card {
    background: #151e33;
    border: 2px solid #26375a;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

.highlight-box {
    border: 2px solid #63e660;
    box-shadow: 0 0 12px rgba(99, 230, 96, 0.4);
    background: rgba(21, 30, 51, 0.8);
}

.slider-gradient {
    background: linear-gradient(to right, #fbd835, #63e660, #21cbd3, #f44090);
    height: 8px;
    border-radius: 4px;
}

.footer-text {
    font-size: 10px;
    color: #5d7293;
    text-align: center;
    margin-top: 15px;
}
```
v1.0.0 • Kelompok Newton • NLP Final Project