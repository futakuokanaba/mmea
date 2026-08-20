"""
Peta Risiko Wajib Pajak Sektor MMEA — Dashboard Streamlit
Terintegrasi dengan data populasi hasil pemodelan & model Random Forest (.pkl) aktual.

Jalankan dengan:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Peta Risiko MMEA",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mmea_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_rf_mmea_bundle.pkl")

NAVY = "#09264a"
BLUE = "#1677e8"
GREEN = "#38a169"
YELLOW = "#e8b339"
RED = "#e55353"

# ----------------------------------------------------------------------------
# CSS RINGAN AGAR TAMPILAN LEBIH DEKAT DENGAN MOCKUP ASLI
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #f4f7fb; }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY}, #061b35);
    }}
    section[data-testid="stSidebar"] * {{ color: #dbe7f5 !important; }}
    div[data-testid="stMetric"] {{
        background: #ffffff; border: 1px solid #dce4ef; border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 3px 12px rgba(23,43,77,.04);
    }}
    .badge {{ border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 800;
        display:inline-block; margin-right:6px; }}
    .badge.high {{ background:#ffe0e0; color:#c53030; }}
    .badge.mid {{ background:#fff1c7; color:#9b6b00; }}
    .badge.low {{ background:#dcf5e5; color:#26734d; }}
    .info-card {{ background:#fff; border:1px solid #dce4ef; border-radius:12px; padding:16px 18px; height:100%; }}
    .info-card h4 {{ margin-top:0; }}
    .risk-result {{ border-radius:12px; padding:22px; text-align:center; color:#fff; }}
    .risk-result small {{ letter-spacing:.5px; font-weight:700; }}
    .risk-result h1 {{ margin:8px 0; color:#fff; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# DATA & MODEL LOADING (CACHED)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    def level_from_kuadran(k):
        if k in ["X3Y3", "X3Y2", "X2Y3"]:
            return "Tinggi"
        if k in ["X2Y2", "X3Y1", "X1Y3"]:
            return "Sedang"
        return "Rendah"

    df["LEVEL_RISIKO"] = df["KUADRAN"].apply(level_from_kuadran)
    return df


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


def level_from_kuadran(k: str) -> str:
    if k in ["X3Y3", "X3Y2", "X2Y3"]:
        return "Tinggi"
    if k in ["X2Y2", "X3Y1", "X1Y3"]:
        return "Sedang"
    return "Rendah"


def badge_html(level: str) -> str:
    cls = {"Tinggi": "high", "Sedang": "mid", "Rendah": "low"}[level]
    return f'<span class="badge {cls}">{level}</span>'


def fmt_rp(v: float) -> str:
    if v >= 1e9:
        return f"Rp {v/1e9:,.2f} M".replace(",", ".")
    if v >= 1e6:
        return f"Rp {v/1e6:,.1f} Jt".replace(",", ".")
    return f"Rp {v:,.0f}".replace(",", ".")


df = load_data()
bundle = load_model_bundle()

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGASI
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='display:flex;gap:10px;align-items:center;padding:5px 0 22px'>"
    "<div style='width:38px;height:38px;border-radius:10px;background:#ffd43b;"
    "color:#09264a;display:flex;align-items:center;justify-content:center;font-weight:900'>DJP</div>"
    "<div style='font-weight:800;font-size:15px'>PETA RISIKO<br>MMEA</div></div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Landing Page", "📊 List Populasi", "🎯 Model Prediction", "📄 Dokumentasi"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    "<div style='margin-top:25px;color:#b9c8dc;font-size:12px;line-height:1.6;"
    "border-top:1px solid #1a3a5e;padding-top:14px'>Sumber: hasil pemodelan aktual "
    "(notebook + populasi kuadran)<br>© 2026 Peta Risiko MMEA<br>"
    "Direktorat Jenderal Pajak</div>",
    unsafe_allow_html=True,
)

TOTAL_OBS = len(df)
TOTAL_WP = df["NPWP9_Masking"].nunique()

# ============================================================================
# HALAMAN 1 — LANDING PAGE
# ============================================================================
if page == "🏠 Landing Page":
    st.title("Peta Risiko Wajib Pajak Sektor MMEA")
    st.caption("Dashboard analitik terintegrasi — data populasi & model hasil pemodelan aktual")

    with st.container():
        c1, c2 = st.columns([1.7, 1])
        with c1:
            st.markdown("## Peta Risiko Wajib Pajak Sektor Minuman Mengandung Etil Alkohol (MMEA)")
            st.markdown(
                """
                Dashboard ini menggabungkan **risiko kemungkinan (Sumbu X)** hasil model klasifikasi
                Random Forest berbasis 7 variabel profil, dan **risiko dampak (Sumbu Y)** berbasis nilai
                rupiah selisih penjualan/omzet, untuk mendukung prioritas pengawasan dan pemeriksaan.
                Seluruh angka pada dashboard ini diambil langsung dari output populasi hasil pemodelan.
                """
            )
            pct_badan = (df["JENIS_WP"].eq("BADAN").mean() * 100)
            st.markdown(
                f'<span class="badge mid">POPULASI {TOTAL_WP} WP PRODUSEN MMEA</span>'
                f'<span class="badge low">{TOTAL_OBS} OBSERVASI WP-TAHUN</span>'
                f'<span class="badge low">PERIODE 2022–2024</span>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<div style='font-size:90px;text-align:center'>🍾🍷🍺🥃</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class="info-card"><h4>▢ Latar Belakang</h4>
            <p style="font-size:13px;color:#52657e;line-height:1.6">
            Industri MMEA memiliki karakteristik rantai pasok, penjualan, produksi, importasi,
            dan distribusi yang kompleks. Model risiko dibangun dari data historis WP-Tahun untuk
            mengidentifikasi kombinasi risiko kemungkinan dan dampak yang tinggi. 
            Diperlukan tools pengawasan WP sektor MMEA untuk mendukung fungsi regulerend.</p></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        pct_op = 100 - pct_badan
        st.markdown(
            f"""<div class="info-card"><h4>🍾 Gambaran Industri MMEA</h4>
            <ul style="font-size:13px;color:#52657e;line-height:1.7">
            <li>Populasi: {TOTAL_WP} perusahaan produsen MMEA</li>
            <li>Jenis WP: Badan {pct_badan:.1f}%, Orang Pribadi {pct_op:.1f}%</li>
            <li>Dipengaruhi ketentuan cukai dan PPN</li>
            <li>Risiko: underreporting penjualan, manipulasi biaya, mismatch data cukai</li>
            </ul></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="info-card"><h4>▤ Data & Model yang Digunakan</h4>
            <ul style="font-size:13px;color:#52657e;line-height:1.7">
            <li>{TOTAL_WP} Wajib Pajak Produsen MMEA</li>
            <li>{TOTAL_OBS} observasi WP-Tahun (2022–2024)</li>
            <li>7 variabel X (kemungkinan) + 2 variabel Y (dampak)</li>
            <li>Target/label: <code>D_001</code> · Model final: <b>Random Forest</b></li>
            </ul></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(
            """<div class="info-card"><h4>🧑‍💼 Susunan Tim</h4>
            <ul style="font-size:13px;color:#52657e;line-height:1.7">
            <li>Business Leader: Aji M. Elvin Nor</li><li>Business Analyst: Andriani Saparingsih</li><li>Data Engineer: Nabila Ghina Naufalita</li>
            <li>Data Scientist - Statistical Modelling: Tasiman</li><li>Data Scientist - Visualization: Dwi Purwanto</li></ul></div>""",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            """<div class="info-card"><h4>⚠ Tantangan</h4>
            <ul style="font-size:13px;color:#52657e;line-height:1.7">
            <li>Kualitas dan kelengkapan data antar tahun</li>
            <li>Variasi skala nilai rupiah (outlier ekstrem)</li>
            <li>Ketidakseimbangan distribusi label (94 vs 74 setelah balancing)</li>
            <li>Interpretabilitas model Random Forest</li></ul></div>""",
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            """<div class="info-card"><h4>◎ Tujuan Project</h4>
            <ul style="font-size:13px;color:#52657e;line-height:1.7">
            <li>Mengembangkan peta risiko MMEA dari data riil hasil pemodelan</li>
            <li>Mengelompokkan WP ke 9 kuadran (X1–3 × Y1–3) → level Rendah/Sedang/Tinggi</li>
            <li>Mendukung prioritas pengawasan/pemeriksaan berbasis skor</li></ul></div>""",
            unsafe_allow_html=True,
        )

# ============================================================================
# HALAMAN 2 — LIST POPULASI
# ============================================================================
elif page == "📊 List Populasi":
    st.title("List Populasi & Statistik")
    st.caption(f"Eksplorasi populasi Wajib Pajak MMEA periode 2022–2024 — data aktual hasil pemodelan (n={TOTAL_OBS})")

    year_options = ["Semua (2022–2024)"] + sorted(df["TAHUN_PETA"].unique().tolist())
    year_filter = st.selectbox("Filter Tahun (Ringkasan)", year_options, index=0)
    df_view = df if year_filter == "Semua (2022–2024)" else df[df["TAHUN_PETA"] == year_filter]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Wajib Pajak Produsen MMEA", f"{TOTAL_WP}", "NPWP unik")
    k2.metric("Total Observasi", f"{TOTAL_OBS}", "WP-Tahun")
    yc = df["TAHUN_PETA"].value_counts().sort_index()
    k3.metric("Periode", "2022–2024", " / ".join(str(v) for v in yc.values))
    k4.metric("Variabel", "10", "7 X + 2 Y + 1 Target (D_001)")

    tab1, tab2, tab3 = st.tabs(["Ringkasan", "List Populasi", "Statistik Populasi"])

    # ---------------- TAB RINGKASAN ----------------
    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Sebaran Populasi WP-Tahun Menurut Level Risiko")
            level_counts = df_view["LEVEL_RISIKO"].value_counts()
            order = ["Tinggi", "Sedang", "Rendah"]
            values = [level_counts.get(l, 0) for l in order]
            colors = [RED, YELLOW, GREEN]
            fig = go.Figure(
                go.Funnel(
                    y=[f"{l} ({v} · {v/len(df_view)*100:.1f}%)" for l, v in zip(order, values)],
                    x=values,
                    marker={"color": colors},
                    textinfo="value",
                )
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Level risiko = pemetaan 9 kuadran (KUADRAN_X × KUADRAN_Y): "
                "Tinggi = X3Y3/X3Y2/X2Y3, Sedang = X2Y2/X3Y1/X1Y3, Rendah = sisanya."
            )
        with colB:
            st.subheader("Top 10 Wajib Pajak Paling Berisiko (Kuadran X3Y3)")
            top10 = (
                df_view[df_view["KUADRAN"] == "X3Y3"]
                .sort_values(["SKOR_Y", "SKOR_X"], ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            top10_disp = top10[["NPWP9_Masking", "TAHUN_PETA", "NAMA_WP_Masking", "SKOR_X", "SKOR_Y", "LEVEL_RISIKO"]].copy()
            top10_disp.index += 1
            top10_disp["SKOR_X"] = top10_disp["SKOR_X"].round(2)
            top10_disp["SKOR_Y"] = top10_disp["SKOR_Y"].apply(fmt_rp)
            top10_disp.columns = ["NPWP", "Tahun", "Nama WP", "Skor X", "Skor Y", "Level"]
            st.dataframe(top10_disp, use_container_width=True, height=320)
            st.download_button(
                "⬇ Download Top 10 (CSV)",
                data=top10.to_csv(index=False).encode("utf-8"),
                file_name="top10_risiko_mmea.csv",
                mime="text/csv",
            )

        colC, colD = st.columns(2)
        with colC:
            st.subheader("Matriks Kuadran Risiko (KUADRAN_X × KUADRAN_Y)")
            pivot = df_view.pivot_table(
                index="KUADRAN_Y", columns="KUADRAN_X", values="NPWP9_Masking", aggfunc="count", fill_value=0
            ).reindex(index=[3, 2, 1], columns=[1, 2, 3], fill_value=0)
            z = pivot.values
            level_grid = [[level_from_kuadran(f"X{x}Y{y}") for x in [1, 2, 3]] for y in [3, 2, 1]]
            color_map = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
            zc = [[color_map[v] for v in row] for row in level_grid]
            fig2 = go.Figure(
                data=go.Heatmap(
                    z=zc,
                    text=z,
                    texttemplate="%{text}",
                    x=["Rendah (X1)", "Sedang (X2)", "Tinggi (X3)"],
                    y=["Tinggi (Y3)", "Sedang (Y2)", "Rendah (Y1)"],
                    colorscale=[[0, GREEN], [0.5, YELLOW], [1, RED]],
                    showscale=False,
                    textfont={"size": 16, "color": "white"},
                )
            )
            fig2.update_layout(
                margin=dict(l=10, r=10, t=10, b=30), height=300,
                xaxis_title="Risiko Kemungkinan (X) →", yaxis_title="↑ Risiko Dampak (Y)",
            )
            st.plotly_chart(fig2, use_container_width=True)
        with colD:
            st.subheader("Komposisi Wajib Pajak Berdasarkan Jenis")
            jc = df_view["JENIS_WP"].value_counts().reset_index()
            jc.columns = ["Jenis WP", "Jumlah"]
            fig3 = px.pie(jc, names="Jenis WP", values="Jumlah", hole=0.55, color_discrete_sequence=[BLUE, "#58a77b"])
            fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig3, use_container_width=True)

    # ---------------- TAB LIST POPULASI ----------------
    with tab2:
        f1, f2, f3, f4 = st.columns(4)
        search = f1.text_input("Cari NPWP / Nama WP")
        fy = f2.selectbox("Tahun", ["Semua"] + sorted(df["TAHUN_PETA"].unique().tolist()))
        fj = f3.selectbox("Jenis WP", ["Semua", "BADAN", "OP NON KARYAWAN"])
        fl = f4.selectbox("Level Risiko", ["Semua", "Tinggi", "Sedang", "Rendah"])

        filtered = df.copy()
        if search:
            mask = (
                filtered["NPWP9_Masking"].str.contains(search, case=False, na=False)
                | filtered["NAMA_WP_Masking"].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if fy != "Semua":
            filtered = filtered[filtered["TAHUN_PETA"] == fy]
        if fj != "Semua":
            filtered = filtered[filtered["JENIS_WP"] == fj]
        if fl != "Semua":
            filtered = filtered[filtered["LEVEL_RISIKO"] == fl]

        st.caption(f"{len(filtered)} baris ditemukan")
        show_cols = [
            "NPWP9_Masking", "TAHUN_PETA", "NAMA_WP_Masking", "NAMA_KPP", "JENIS_WP",
            "SKOR_X", "SKOR_Y", "KUADRAN", "LEVEL_RISIKO",
        ]
        st.dataframe(filtered[show_cols], use_container_width=True, height=480)
        st.download_button(
            "⬇ Download hasil filter (CSV)",
            data=filtered[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="populasi_mmea_filtered.csv",
            mime="text/csv",
        )

    # ---------------- TAB STATISTIK ----------------
    with tab3:
        colE, colF = st.columns(2)
        with colE:
            st.subheader(f"Statistik Deskriptif Variabel (n={TOTAL_OBS})")
            var_cols = ["X_001", "X_002", "X_003", "X_004", "X_005", "X_006", "X_007", "Y_001", "Y_002"]
            labels = {"X_001": "X1", "X_002": "X2", "X_003": "X3", "X_004": "X4", "X_005": "X5",
                      "X_006": "X6", "X_007": "X7", "Y_001": "Y1 (Rp)", "Y_002": "Y2 (Rp)"}
            stat_rows = []
            for c in var_cols:
                s = df[c]
                is_rp = c.startswith("Y")
                fmt = fmt_rp if is_rp else (lambda x: f"{x:.3f}")
                stat_rows.append(
                    {"Variabel": labels[c], "Mean": fmt(s.mean()), "Median": fmt(s.median()),
                     "Std Dev": fmt(s.std()), "Min": fmt(s.min()), "Max": fmt(s.max())}
                )
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
            st.caption("Nilai X_001–X_007 dalam skala asli (belum di-scaling); Y_001, Y_002 dalam Rupiah.")
        with colF:
            st.subheader("Distribusi Observasi per Tahun")
            yc2 = df["TAHUN_PETA"].value_counts().sort_index().reset_index()
            yc2.columns = ["Tahun", "Jumlah"]
            yc2["% dari Total"] = (yc2["Jumlah"] / TOTAL_OBS * 100).round(1).astype(str) + "%"
            st.dataframe(yc2, use_container_width=True, hide_index=True)

            st.subheader("Distribusi Kuadran (9 Kombinasi)")
            kc = df["KUADRAN"].value_counts().sort_index().reset_index()
            kc.columns = ["Kuadran", "Jumlah"]
            kc["% dari Total"] = (kc["Jumlah"] / TOTAL_OBS * 100).round(1).astype(str) + "%"
            st.dataframe(kc, use_container_width=True, hide_index=True)

# ============================================================================
# HALAMAN 3 — MODEL PREDICTION
# ============================================================================
elif page == "🎯 Model Prediction":
    st.title("Model Prediction")
    st.caption("Masukkan profil dan data dampak WP baru untuk memperoleh estimasi level risiko")
    st.info(
        "Skor dihasilkan langsung dari model **Random Forest terlatih** "
        "(`model_rf_mmea_bundle.pkl`) — bukan pendekatan/aproksimasi.",
        icon="✅",
    )

    model = bundle["model"]
    feature_order = bundle["feature_order"]
    scaler_params = bundle["scaler_params"]
    lo, hi = bundle["scaler_feature_range"]
    importance = bundle["feature_importance"]
    qx_th = bundle["quadran_x_thresholds"]
    qy_th = bundle["quadran_y_thresholds"]

    def scale_minmax(value, feat):
        dmin = scaler_params[feat]["data_min_"]
        dmax = scaler_params[feat]["data_max_"]
        if dmax == dmin:
            return lo
        v = max(dmin, min(dmax, value))
        return lo + (v - dmin) / (dmax - dmin) * (hi - lo)

    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        st.subheader("Input Data")
        with st.form("predict_form"):
            x1 = st.selectbox("X1 — Penjualan ke NPPBKC non Penyalur", ["Ya (1)", "Tidak (0)"], index=0)
            x2 = st.selectbox("X2 — NPPBKC aktif tanpa transaksi CK-5", ["Ya (1)", "Tidak (0)"], index=0)
            x3 = st.selectbox("X3 — NPWP tidak ada transaksi penjualan cukai", ["Ya (1)", "Tidak (0)"], index=1)
            x4 = st.number_input("X4 — Rasio kuantitas produksi (rentang data: 0.00–4.16)", min_value=0.0, value=0.10, step=0.01, format="%.2f")
            x5 = st.number_input("X5 — Kewajiban tidak lapor SPT / omzet nihil (rentang data: 0.00–315.59)", min_value=0.0, value=1.00, step=0.01, format="%.2f")
            x6 = st.number_input("X6 — Rasio GPM (rentang data: 0.00–3.42 · bobot terbesar 34%)", min_value=0.0, value=0.70, step=0.01, format="%.2f")
            x7 = st.number_input("X7 — Rasio Kinerja Pajak ETR (rentang data: 0.00–0.22)", min_value=0.0, value=0.15, step=0.01, format="%.2f")
            y1 = st.number_input("Y1 — Selisih harga jual eceran (Rp)", min_value=0, value=25_000_000, step=1_000_000)
            y2 = st.number_input("Y2 — Selisih omzet pada SPT (Rp)", min_value=0, value=10_500_000, step=1_000_000)
            submitted = st.form_submit_button("▶ Prediksi Risiko", use_container_width=True)

    with col_result:
        if submitted:
            raw_input = {
                "X_001": 1 if x1.startswith("Ya") else 0,
                "X_002": 1 if x2.startswith("Ya") else 0,
                "X_003": 1 if x3.startswith("Ya") else 0,
                "X_004": x4, "X_005": x5, "X_006": x6, "X_007": x7,
            }
            scaled_row = {f: scale_minmax(raw_input[f], f) for f in feature_order}
            X_df = pd.DataFrame([scaled_row], columns=feature_order)
            skor_x = float(model.predict_proba(X_df)[0][1])
            skor_y = max(y1, y2)

            kx = 1 if skor_x < qx_th["low"] else (3 if skor_x > qx_th["high"] else 2)
            ky = 1 if skor_y <= qy_th["low"] else (3 if skor_y > qy_th["high"] else 2)
            kuadran = f"X{kx}Y{ky}"
            level = level_from_kuadran(kuadran)
            level_color = {"Tinggi": RED, "Sedang": YELLOW, "Rendah": GREEN}[level]

            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.markdown(
                    f"""<div class="risk-result" style="background:{level_color}">
                    <small>LEVEL RISIKO</small><h1>{level.upper()}</h1>
                    <div>Skor X (Kemungkinan) <b>{skor_x:.2f} / 1.00</b></div>
                    <div>Kuadran <b>{kuadran}</b></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with rcol2:
                st.markdown("**Posisi pada Peta Risiko**")
                level_grid = [[level_from_kuadran(f"X{x}Y{y}") for x in [1, 2, 3]] for y in [3, 2, 1]]
                color_map = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
                zc = [[color_map[v] for v in row] for row in level_grid]
                marker_text = [
                    ["★" if (x == kx and (3 - i) == ky) else "" for i, x in enumerate([x for x in [1, 2, 3]])]
                    for _ in [0]
                ]
                text_grid = []
                for yy in [3, 2, 1]:
                    row_txt = []
                    for xx in [1, 2, 3]:
                        label = f"X{xx}Y{yy}"
                        row_txt.append(f"★ {label}" if (xx == kx and yy == ky) else label)
                    text_grid.append(row_txt)
                figp = go.Figure(
                    data=go.Heatmap(
                        z=zc, text=text_grid, texttemplate="%{text}",
                        x=["Rendah (X1)", "Sedang (X2)", "Tinggi (X3)"],
                        y=["Tinggi (Y3)", "Sedang (Y2)", "Rendah (Y1)"],
                        colorscale=[[0, GREEN], [0.5, YELLOW], [1, RED]],
                        showscale=False, textfont={"size": 13, "color": "white"},
                    )
                )
                figp.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)
                st.plotly_chart(figp, use_container_width=True)

            st.markdown("**Kontribusi Variabel (Bobot Random Forest)**")
            contrib_df = pd.DataFrame(
                {"Variabel": [k.replace("X_00", "X") for k in feature_order],
                 "Bobot Importance": [importance[k] for k in feature_order],
                 "Nilai (scaled)": [scaled_row[k] for k in feature_order]}
            )
            contrib_df["Kontribusi"] = contrib_df["Bobot Importance"] * contrib_df["Nilai (scaled)"]
            contrib_df = contrib_df.sort_values("Kontribusi", ascending=True)
            figc = px.bar(contrib_df, x="Kontribusi", y="Variabel", orientation="h", color_discrete_sequence=[BLUE])
            figc.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)
            st.plotly_chart(figc, use_container_width=True)

            st.markdown("**Rekomendasi**")
            if level == "Tinggi":
                st.success(
                    "✔ Prioritaskan untuk pengawasan dan pemeriksaan.\n\n"
                    "✔ Lakukan analisis lanjutan atas variabel berdampak tinggi.\n\n"
                    "✔ Periksa kepatuhan terkait pelaporan penjualan dan biaya."
                )
            elif level == "Sedang":
                st.warning(
                    "✔ Masukkan dalam daftar pengawasan berkala.\n\n"
                    "✔ Lakukan konfirmasi data ke WP terkait variabel dominan.\n\n"
                    "✔ Pantau perkembangan skor pada periode berikutnya."
                )
            else:
                st.info(
                    "✔ Prioritas rendah untuk pemeriksaan saat ini.\n\n"
                    "✔ Cukup dilakukan edukasi/pelayanan rutin.\n\n"
                    "✔ Tetap pantau bila ada perubahan data signifikan."
                )

            with st.expander("Penjelasan Model"):
                st.write(
                    f"SKOR_X = **{skor_x:.4f}** dihasilkan langsung dari "
                    f"`model.predict_proba()` model Random Forest terlatih "
                    f"(akurasi test set 97,06%). SKOR_Y = maks(Y1, Y2) = {fmt_rp(skor_y)}. "
                    "Kuadran ditentukan dari threshold aktual: KUADRAN_X "
                    "(1: <0,01 · 2: 0,01–0,50 · 3: >0,50), KUADRAN_Y "
                    "(1: ≤50 juta · 2: 50 juta–1 miliar · 3: >1 miliar)."
                )
        else:
            st.markdown("Isi data pada form di sebelah kiri lalu klik **Prediksi Risiko**.")

# ============================================================================
# HALAMAN 4 — DOKUMENTASI
# ============================================================================
elif page == "📄 Dokumentasi":
    st.title("Dokumentasi Metodologi")
    st.caption("Ringkasan proses pemodelan aktual (dari notebook pemodelan MMEA 2026)")

    colL, colR = st.columns(2)
    with colL:
        st.subheader("Alur Proses Pemodelan")
        steps = [
            ("Data & Preprocessing", "Data sumber: 1 file populasi WP-Tahun (2022–2024). Tipe data dirapikan, dilakukan pengecekan null, nol, nilai negatif, dan duplikasi NPWP-Tahun."),
            ("Total Hit X & Y", "Dihitung jumlah variabel X dan Y yang ter-hit (>0) per WP untuk menyaring populasi yang relevan dimodelkan pada Sumbu X."),
            ("Treatment Outlier", "Deteksi outlier variabel X_001–X_007 menggunakan Z-score, nilai di luar ±3 standar deviasi di-cap ke batas atas/bawah data non-outlier."),
            ("Min-Max Scaling", "Variabel X_001–X_007 di-scale ke rentang 0,01–1,00 menggunakan MinMaxScaler sebelum masuk ke pemodelan."),
            ("Uji Signifikansi (ANOVA)", "Uji One-way ANOVA per variabel X terhadap label kepatuhan (LABEL_RASIO) untuk memeriksa signifikansi (p < 0,1)."),
            ("Balancing & Split", "Dari 557 WP-Tahun berlabel D_001, label 0 di-undersample dari 483 → 94 (mendekati jumlah label 1 = 74), total 168 data, di-split 80:20 → train 134, test 34."),
            ("Perbandingan Model", "5 algoritma klasifikasi diuji: LightGBM, Logistic Regression, Random Forest, CatBoost, XGBoost."),
            ("Pemilihan Model & Skoring", "Random Forest dipilih sebagai model final (False Negative = 0, seluruh kasus positif terdeteksi). predict_proba dari model ini menjadi SKOR_X untuk seluruh populasi."),
            ("Skor Dampak (Sumbu Y)", "SKOR_Y = maks(Y_001, Y_002) — selisih harga jual eceran dan selisih omzet SPT, dalam Rupiah."),
            ("Pemetaan Kuadran", "KUADRAN_X dan KUADRAN_Y ditentukan dari threshold, lalu digabung menjadi 9 kuadran (X1Y1…X3Y3) sebagai output populasi akhir."),
        ]
        for i, (title, desc) in enumerate(steps, start=1):
            st.markdown(f"**{i}. {title}**")
            st.caption(desc)

    with colR:
        st.subheader("Perbandingan Performa Model (Test Set, n=34)")
        model_perf = pd.DataFrame(
            {
                "Model": ["Random Forest ⭐ (Final)", "LightGBM", "XGBoost", "Logistic Regression", "CatBoost"],
                "Accuracy": ["97,06%", "97,06%", "97,06%", "94,12%", "94,12%"],
                "Precision": ["97,22%", "94,44%", "97,22%", "94,12%", "94,12%"],
                "Recall": ["97,06%", "100,00%", "97,06%", "94,12%", "94,12%"],
                "F1-score": ["97,06%", "97,14%", "97,06%", "94,12%", "94,12%"],
            }
        )
        st.dataframe(model_perf, use_container_width=True, hide_index=True)
        st.caption(
            "Random Forest dipilih meski akurasi setara LightGBM/XGBoost, karena menghasilkan "
            "False Negative = 0 pada confusion matrix — seluruh WP tidak patuh pada data test "
            "berhasil terdeteksi."
        )

        st.subheader("Feature Importance — Random Forest (Model Final)")
        imp_df = pd.DataFrame(
            {"Variabel": [k.replace("X_00", "X") for k in bundle["feature_importance"].keys()],
             "Importance": list(bundle["feature_importance"].values())}
        ).sort_values("Importance", ascending=True)
        figimp = px.bar(imp_df, x="Importance", y="Variabel", orientation="h", color_discrete_sequence=[BLUE])
        figimp.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)
        st.plotly_chart(figimp, use_container_width=True)

        st.subheader("Aturan Threshold Kuadran")
        th_df = pd.DataFrame(
            {
                "Aturan": ["KUADRAN_X = 1", "KUADRAN_X = 2", "KUADRAN_X = 3",
                           "KUADRAN_Y = 1", "KUADRAN_Y = 2", "KUADRAN_Y = 3"],
                "Kondisi": ["SKOR_X < 0,01", "0,01 ≤ SKOR_X ≤ 0,50", "SKOR_X > 0,50",
                            "SKOR_Y ≤ Rp 50 juta", "Rp 50 juta < SKOR_Y ≤ Rp 1 miliar", "SKOR_Y > Rp 1 miliar"],
            }
        )
        st.dataframe(th_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Peta Risiko MMEA • Dashboard analitik • Data aktual dari file populasi hasil pemodelan (KUADRAN_MMEA_DAS19082026_Masking)")
