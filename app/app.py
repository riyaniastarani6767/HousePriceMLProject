import streamlit as st
import pandas as pd
from pathlib import Path

from src.data_preprocessing import read_superstore_csv
from src.feature_engineering import add_basic_features
# --- add project root to sys.path ---
import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # parent of /app
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ----------------------------------

st.set_page_config(page_title="Superstore Dashboard", page_icon="🏪", layout="wide")

@st.cache_data(show_spinner=False)
def load_default_data():
    path = "data/raw/USSuperstoreData.csv"
    if Path(path).exists():
        df = read_superstore_csv(path)
        df = add_basic_features(df)
        return df
    return pd.DataFrame()

st.title("🏪 Superstore Analytics & ML Dashboard")
st.write("Selamat datang! Gunakan sidebar untuk **upload CSV** (opsional).")

with st.sidebar:
    st.subheader("📥 Data Source")
    uploaded = st.file_uploader("Upload CSV Superstore (separator ;, desimal ,)", type=["csv"])
    if uploaded:
        df = read_superstore_csv(uploaded)
        df = add_basic_features(df)
        st.session_state["data_df"] = df
        st.success(f"Data loaded: {len(df):,} rows")
    else:
        if "data_df" not in st.session_state:
            st.session_state["data_df"] = load_default_data()
        st.info("Using default data from `data/raw/USSuperstoreData.csv`")

st.write("👉 Buka halaman **EDA Analysis** atau **Profit Prediction** di sidebar kiri (Pages).")
if len(st.session_state["data_df"]) == 0:
    st.warning("Belum ada data. Silakan upload CSV di sidebar atau letakkan file default di `data/raw/USSuperstoreData.csv`.")
else:
    st.caption(f"Rows: {len(st.session_state['data_df']):,}")
