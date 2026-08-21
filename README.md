# Peta Risiko MMEA — Versi Streamlit

Versi Streamlit dari dashboard "Peta Risiko Wajib Pajak Sektor MMEA", terintegrasi dengan
data populasi hasil pemodelan dan model Random Forest terlatih aktual.

## Struktur folder

```
streamlit_app/
├── app.py                          # aplikasi utama (4 halaman)
├── requirements.txt
├── data/
│   └── mmea_data.csv               # data populasi 661 observasi WP-Tahun (2022–2024)
└── model/
    └── model_rf_mmea_bundle.pkl    # model Random Forest + scaler + metadata
```

Jangan pindahkan `app.py` tanpa ikut memindahkan folder `data/` dan `model/` di sebelahnya —
path dibaca relatif terhadap lokasi `app.py`.

## Cara menjalankan

```bash
python -m venv venv
venv\Scripts\activate #source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Lalu buka `http://localhost:8501` di browser.
Streamlit ini dapat juga diakses online melalui https://crmmmea.streamlit.app/

## Halaman yang tersedia

1. **Landing Page** — ringkasan proyek, latar belakang, tim, tantangan, tujuan.
2. **List Populasi** — KPI, piramida level risiko, top 10 WP paling berisiko, matriks 9
   kuadran, komposisi jenis WP, tabel populasi lengkap (dengan pencarian & filter), dan
   statistik deskriptif — semua dihitung langsung dari `data/mmea_data.csv`.
3. **Model Prediction** — form input 7 variabel X (X1–X3 boolean Ya/Tidak, X4–X7 numerik)
   dan 2 variabel Y, lalu skor dihasilkan **langsung dari model Random Forest terlatih**
   (`model.predict_proba()`), bukan pendekatan/simulasi.
4. **Dokumentasi** — alur metodologi pemodelan, perbandingan performa 5 model, feature
   importance, dan aturan threshold kuadran.

## Catatan tentang model

`model_rf_mmea_bundle.pkl` direproduksi dari notebook pemodelan asli menggunakan data
populasi (`X_001`–`X_007` dalam skala mentah + label `D_001`), dengan pipeline yang identik:
filter `TOTAL_HIT_X > 0` → capping outlier Z-score (±3 std) → `MinMaxScaler(0.01, 1.0)` →
undersampling kelas mayoritas → split 80:20 (`random_state=50`) →
`RandomForestClassifier(max_depth=8, n_estimators=300, random_state=6847)`.

Karena file Excel mentah asli tidak tersedia, hasil replikasi ini *mendekati* tapi belum
tentu 100% identik dengan model yang dipakai untuk menghasilkan populasi resmi. Untuk
produksi, disarankan melatih ulang dari file sumber asli jika tersedia.
