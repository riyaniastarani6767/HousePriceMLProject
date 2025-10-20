

import streamlit as st
import pandas as pd
from pathlib import Path

# Pastikan paket 'src' bisa diimpor saat Streamlit run dari folder /app
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # parent of /app
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data_preprocessing import read_superstore_csv
from src.feature_engineering import add_basic_features

st.set_page_config(page_title="Superstore Dashboard", layout="wide")

# --------------------------- Komponen UI --------------------------------------
def kpi_card(title: str, value: str, bg: str = "#F1F1F1", fg: str = "#1F2937"):
    """Kartu KPI sederhana dengan warna latar dan teks kustom."""
    html = f"""
    <div style="
        background:{bg};
        color:{fg};
        padding:16px 18px;
        border-radius:12px;
        border:1px solid rgba(0,0,0,0.06);
        font-family:Inter,system-ui,Arial,sans-serif;
        ">
        <div style="font-size:12px; letter-spacing:.3px; opacity:.85; margin-bottom:6px;">
            {title}
        </div>
        <div style="font-size:28px; font-weight:700; line-height:1;">
            {value}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ------------------------------ DATA LOADER -----------------------------------
@st.cache_data(show_spinner=False)
def load_default_data():
    path = "data/raw/USSuperstoreData.csv"
    if Path(path).exists():
        df = read_superstore_csv(path)      # sudah menangani ; dan desimal ,
        df = add_basic_features(df)
        return df
    return pd.DataFrame()

def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    gf = st.session_state.get("global_filters", {})
    out = df.copy()
    if "Order Date" in out.columns and gf.get("date_range"):
        start, end = pd.to_datetime(gf["date_range"][0]), pd.to_datetime(gf["date_range"][1])
        out = out[out["Order Date"].between(start, end)]
    if "Region" in out.columns and gf.get("regions"):
        out = out[out["Region"].isin(gf["regions"])]
    return out

# ------------------------------ STATE SETUP -----------------------------------
if "data_df" not in st.session_state:
    st.session_state["data_df"] = load_default_data()

df_active = st.session_state["data_df"]

# -------------------------------- SIDEBAR -------------------------------------
with st.sidebar:
    st.header("Sumber Data")
    uploaded = st.file_uploader("Upload CSV Superstore (pemisah ;, desimal ,)", type=["csv"])
    if uploaded:
        try:
            df_up = read_superstore_csv(uploaded)
            df_up = add_basic_features(df_up)
            st.session_state["data_df"] = df_up
            df_active = df_up
            st.success(f"Data berhasil dimuat: {len(df_up):,} baris")
        except Exception as e:
            st.error(f"Gagal membaca file. Pastikan pemisah ';' dan desimal ','. Detail: {e}")
    else:
        if df_active.empty:
            st.info("Menggunakan data default: letakkan file di data/raw/USSuperstoreData.csv")

    st.divider()
    st.header("Filter Global")

    if "global_filters" not in st.session_state:
        st.session_state["global_filters"] = {}

    if not df_active.empty and "Order Date" in df_active.columns:
        min_d, max_d = df_active["Order Date"].min(), df_active["Order Date"].max()
        date_range = st.date_input("Rentang Tanggal", value=(min_d, max_d))
        st.session_state["global_filters"]["date_range"] = date_range

    regions_all = sorted(df_active["Region"].dropna().unique().tolist()) if "Region" in df_active else []
    regions_pick = st.multiselect("Region", regions_all, default=regions_all)
    st.session_state["global_filters"]["regions"] = regions_pick

# --------------------------------- MAIN ---------------------------------------
st.title("Superstore Analytics & ML Dashboard")
st.caption("Analisis penjualan, profit, dan prediksi profitabilitas pada dataset Superstore.")

fdf = apply_global_filters(df_active)

# Kartu status data (berwarna)
c1, c2, c3, c4 = st.columns(4)
rows = len(fdf)
orders = fdf["Order ID"].nunique() if "Order ID" in fdf else rows
period = "-"
if "Order Date" in fdf and not fdf.empty:
    period = f"{fdf['Order Date'].min().date()} s.d. {fdf['Order Date'].max().date()}"
source = "Upload" if uploaded else ("Default" if not df_active.empty else "-")

with c1: kpi_card("Total Baris", f"{rows:,}", bg="#E8F1FF", fg="#0F3D91")
with c2: kpi_card("Jumlah Order Unik", f"{orders:,}", bg="#F1F1F1", fg="#333333")
with c3: kpi_card("Periode Data", period, bg="#FFF4E5", fg="#8A4B08")
with c4: kpi_card("Sumber Data", source, bg="#E8FFF2", fg="#0C6B3E")

st.divider()

# Tautan ke halaman lain (path relatif dari folder 'app')
colA, colB, _ = st.columns([1,1,2])
with colA:
    st.page_link("pages/EDA_Analysis.py", label="Buka EDA Analysis")
with colB:
    st.page_link("pages/Profit_Prediction.py", label="Buka Profit Prediction")

# Pratinjau tabel
st.subheader("Pratinjau Data Aktif")
if fdf.empty:
    st.warning("Belum ada data yang bisa ditampilkan. Upload file atau gunakan data default.")
else:
    st.dataframe(fdf.head(100), use_container_width=True, height=420)
    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Unduh data terfilter (CSV)", data=csv_bytes,
                       file_name="superstore_filtered.csv", mime="text/csv")
