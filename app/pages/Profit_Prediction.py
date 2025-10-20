

import streamlit as st
import pandas as pd
from pathlib import Path

# Pastikan paket 'src' bisa diimpor saat Streamlit run dari folder /app
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # parent of /app
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model_utils import load_model

st.set_page_config(page_title="Profit Prediction", layout="wide")

MODEL_PATH = "models/model_profit_clf.joblib"

@st.cache_resource(show_spinner=False)
def get_model():
    if not Path(MODEL_PATH).exists():
        return None
    return load_model(MODEL_PATH)

model = get_model()
if model is None:
    st.error("Model belum tersedia. Jalankan training terlebih dahulu untuk menghasilkan models/model_profit_clf.joblib.")
    st.stop()

df_ref = st.session_state.get("data_df", pd.DataFrame())

# ---------------------------- PREDIKSI TUNGGAL -------------------------------
st.subheader("Prediksi Tunggal")
col_form, col_out = st.columns([1,1])

with col_form:
    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        sales = c1.number_input("Sales", min_value=0.0, value=100.0, step=10.0)
        qty = c2.number_input("Quantity", min_value=1, value=1, step=1)
        disc = c3.number_input("Discount (0–0.9)", min_value=0.0, max_value=0.9, value=0.0, step=0.05)

        c4, c5, c6 = st.columns(3)
        ship_opts = sorted(df_ref["Ship Mode"].dropna().unique()) if "Ship Mode" in df_ref else ["Standard Class","Second Class","First Class","Same Day"]
        seg_opts  = sorted(df_ref["Segment"].dropna().unique())   if "Segment" in df_ref else ["Consumer","Corporate","Home Office"]
        cat_opts  = sorted(df_ref["Category"].dropna().unique())  if "Category" in df_ref else ["Furniture","Office Supplies","Technology"]
        ship_mode = c4.selectbox("Ship Mode", ship_opts)
        segment   = c5.selectbox("Segment", seg_opts)
        category  = c6.selectbox("Category", cat_opts)

        if "Sub-Category" in df_ref and "Category" in df_ref:
            subcat_opts = sorted(df_ref[df_ref["Category"]==category]["Sub-Category"].dropna().unique())
        else:
            subcat_opts = sorted(df_ref["Sub-Category"].dropna().unique()) if "Sub-Category" in df_ref else []
        subcat = st.selectbox("Sub-Category", subcat_opts)

        region_opts = sorted(df_ref["Region"].dropna().unique()) if "Region" in df_ref else ["West","East","Central","South"]
        region = st.selectbox("Region", region_opts)
        days_to_ship = st.number_input("Days_to_Ship", min_value=-1, value=2, step=1,
                                       help="Selisih hari (Ship Date - Order Date). Boleh -1 untuk same-day.")

        submitted = st.form_submit_button("Prediksi")

with col_out:
    if submitted:
        row = pd.DataFrame([{
            "Sales": sales,
            "Quantity": qty,
            "Discount": disc,
            "Ship Mode": ship_mode,
            "Segment": segment,
            "Category": category,
            "Sub-Category": subcat,
            "Region": region,
            "Days_to_Ship": days_to_ship
        }])
        proba = model.predict_proba(row)[0][1]
        pred = int(proba >= 0.5)
        st.markdown("Hasil")
        st.write(f"Label: {'Profitable' if pred==1 else 'Not Profitable'}")
        st.write(f"Probabilitas untung: {proba:.2%}")

st.divider()

# ---------------------------- BATCH PREDICTION -------------------------------
st.subheader("Prediksi Batch (CSV)")

required_cols = ["Sales","Quantity","Discount","Ship Mode","Segment","Category","Sub-Category","Region","Days_to_Ship"]

# Template CSV (header saja + 1 baris contoh; pemisah ; dan desimal ,)
template = ";".join(required_cols) + "\n100,00;1;0,10;Standard Class;Consumer;Technology;Phones;West;2\n"
st.download_button("Unduh Template CSV", data=template.encode("utf-8"),
                   file_name="template_batch_superstore.csv", mime="text/csv")
st.caption("Gunakan pemisah ';' dan desimal ',' sesuai contoh pada baris kedua template.")

batch_file = st.file_uploader("Upload CSV batch (hanya kolom fitur sesuai template)", type=["csv"], key="batch_upload")

if batch_file:
    try:
        # Baca CSV dengan aturan Superstore
        batch_df = pd.read_csv(batch_file, sep=";", decimal=",")
        cols = set(batch_df.columns)

        # Jika Days_to_Ship tidak ada tapi ada tanggal, hitung otomatis
        if "Days_to_Ship" not in cols and {"Order Date", "Ship Date"}.issubset(cols):
            od = pd.to_datetime(batch_df["Order Date"], errors="coerce", dayfirst=True)
            sd = pd.to_datetime(batch_df["Ship Date"], errors="coerce", dayfirst=True)
            batch_df["Days_to_Ship"] = (sd - od).dt.days

        # Validasi minimum kolom fitur
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Kolom wajib tidak lengkap: {missing}")
        else:
            st.markdown("Pratinjau Data Batch")
            st.dataframe(batch_df.head(50), use_container_width=True, height=300)

            # Prediksi
            input_df = batch_df[required_cols].copy()
            preds = model.predict(input_df)
            probas = model.predict_proba(input_df)[:,1]

            out = batch_df.copy()
            out["Pred_Profitable"] = preds
            out["Proba_Profit"] = probas

            st.markdown("Hasil Prediksi")
            st.dataframe(out.head(50), use_container_width=True, height=360)

            st.download_button("Unduh Hasil (CSV)",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name="batch_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Gagal memproses file. Pastikan pemisah ';' dan desimal ','. Detail: {e}")
