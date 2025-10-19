import streamlit as st
import pandas as pd
from pathlib import Path
from src.model_utils import load_model
# --- add project root to sys.path ---
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # => folder project root
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ------------------------------------


st.set_page_config(page_title="Profit Prediction", page_icon="🤖", layout="wide")
st.title("🤖 Profit Prediction")

MODEL_PATH = "models/model_profit_clf.joblib"

@st.cache_resource(show_spinner=False)
def get_model():
    if not Path(MODEL_PATH).exists():
        return None
    return load_model(MODEL_PATH)

model = get_model()
if model is None:
    st.error("Model belum tersedia. Jalankan training dulu (lihat notebook / script training).")
    st.stop()

df = st.session_state.get("data_df", pd.DataFrame())

# Build input form
st.subheader("Single Prediction")
with st.form("pred_form"):
    c1, c2, c3 = st.columns(3)
    sales = c1.number_input("Sales", min_value=0.0, value=100.0, step=10.0)
    qty = c2.number_input("Quantity", min_value=1, value=1, step=1)
    disc = c3.number_input("Discount (0–0.9)", min_value=0.0, max_value=0.9, value=0.0, step=0.05)

    c4, c5, c6 = st.columns(3)
    ship_mode = c4.selectbox("Ship Mode", sorted(df["Ship Mode"].dropna().unique()) if "Ship Mode" in df else ["Standard Class","Second Class","First Class","Same Day"])
    segment = c5.selectbox("Segment", sorted(df["Segment"].dropna().unique()) if "Segment" in df else ["Consumer","Corporate","Home Office"])
    category = c6.selectbox("Category", sorted(df["Category"].dropna().unique()) if "Category" in df else ["Furniture","Office Supplies","Technology"])

    subcat_list = sorted(df[df["Category"]==category]["Sub-Category"].dropna().unique()) if "Sub-Category" in df and "Category" in df else []
    subcat = st.selectbox("Sub-Category", subcat_list if subcat_list else (sorted(df["Sub-Category"].dropna().unique()) if "Sub-Category" in df else []))

    region = st.selectbox("Region", sorted(df["Region"].dropna().unique()) if "Region" in df else ["West","East","Central","South"])
    days_to_ship = st.number_input("Days_to_Ship", min_value=-1, value=2, step=1, help="Selisih hari pengiriman (boleh -1 jika same-day)")

    submitted = st.form_submit_button("Prediksi")

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
    st.success(f"Hasil: **{'Profitable' if pred==1 else 'Not Profitable'}**  |  Probabilitas untung: **{proba:.2%}**")

st.divider()
st.subheader("Batch Prediction (CSV)")

batch_file = st.file_uploader("Upload CSV dengan kolom fitur yang sama (tanpa Profit)", type=["csv"], key="batch")
if batch_file:
    batch_df = pd.read_csv(batch_file)
    # Basic guard
    required_cols = ["Sales","Quantity","Discount","Ship Mode","Segment","Category","Sub-Category","Region","Days_to_Ship"]
    missing = [c for c in required_cols if c not in batch_df.columns]
    if missing:
        st.error(f"Kolom kurang: {missing}")
    else:
        preds = model.predict(batch_df)
        probas = model.predict_proba(batch_df)[:,1]
        out = batch_df.copy()
        out["Pred_Profitable"] = preds
        out["Proba_Profit"] = probas
        st.dataframe(out.head(25), use_container_width=True)
        # allow download
        st.download_button("Download Hasil CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="batch_predictions.csv", mime="text/csv")
