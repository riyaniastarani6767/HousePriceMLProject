
import streamlit as st
import pandas as pd

# Pastikan paket 'src' bisa diimpor saat Streamlit run dari folder /app
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # parent of /app
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.visualization import (
    kpi_summary, bar_sales_profit_by_category,
    top_subcategory_by_sales, monthly_trend, scatter_sales_profit
)

st.set_page_config(page_title="EDA Analysis", layout="wide")

# --------------------------- Komponen UI --------------------------------------
def kpi_card(title: str, value: str, bg: str = "#F1F1F1", fg: str = "#1F2937"):
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

st.title("Analisis Bisnis (EDA)")

df = st.session_state.get("data_df", pd.DataFrame())
if df.empty:
    st.warning("Tidak ada data. Kembali ke Beranda untuk upload atau gunakan data default.")
    st.stop()

# Terapkan filter global dari Beranda
gf = st.session_state.get("global_filters", {})
fdf = df.copy()
if "Order Date" in fdf.columns and gf.get("date_range"):
    start, end = pd.to_datetime(gf["date_range"][0]), pd.to_datetime(gf["date_range"][1])
    fdf = fdf[fdf["Order Date"].between(start, end)]
if "Region" in fdf.columns and gf.get("regions"):
    fdf = fdf[fdf["Region"].isin(gf["regions"])]

# Filter khusus halaman (cascade Category -> Sub-Category)
with st.sidebar:
    st.header("Filter Halaman")
    cat_all = sorted(fdf["Category"].dropna().unique().tolist()) if "Category" in fdf else []
    cat_pick = st.multiselect("Category", cat_all, default=cat_all)

    if cat_pick and "Category" in fdf and "Sub-Category" in fdf:
        subcat_all = sorted(fdf[fdf["Category"].isin(cat_pick)]["Sub-Category"].dropna().unique().tolist())
    elif "Sub-Category" in fdf:
        subcat_all = sorted(fdf["Sub-Category"].dropna().unique().tolist())
    else:
        subcat_all = []

    subcat_pick = st.multiselect("Sub-Category", subcat_all, default=subcat_all)

    reset = st.button("Reset Filter Halaman")

if reset:
    st.rerun()

mask = pd.Series(True, index=fdf.index)
if cat_pick and "Category" in fdf:
    mask &= fdf["Category"].isin(cat_pick)
if subcat_pick and "Sub-Category" in fdf:
    mask &= fdf["Sub-Category"].isin(subcat_pick)
fdf = fdf[mask].copy()

# KPI
kpi = kpi_summary(fdf)
c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Total Sales", f"${kpi['total_sales']:,.2f}", bg="#E8F1FF", fg="#0F3D91")
with c2: kpi_card("Total Profit", f"${kpi['total_profit']:,.2f}", bg="#E8FFF2", fg="#0C6B3E")
with c3: kpi_card("Order Unik", f"{kpi['n_orders']:,}", bg="#F1F1F1", fg="#333333")
with c4: kpi_card("Profit Ratio", f"{kpi['profit_ratio']:.2f}%" if kpi['profit_ratio'] is not None else "-", bg="#FFF4E5", fg="#8A4B08")

st.divider()

# Layout dua kolom padat
left, right = st.columns(2)

with left:
    fig1 = bar_sales_profit_by_category(fdf)
    if fig1 is not None:
        st.subheader("Kinerja per Kategori")
        st.plotly_chart(fig1, use_container_width=True)

    fig2 = top_subcategory_by_sales(fdf, top_n=10)
    if fig2 is not None:
        st.subheader("Top 10 Sub-Category berdasarkan Sales")
        st.plotly_chart(fig2, use_container_width=True)

with right:
    fig3 = monthly_trend(fdf)
    if fig3 is not None:
        st.subheader("Tren Bulanan Sales dan Profit")
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = scatter_sales_profit(fdf)
    if fig4 is not None:
        st.subheader("Sebaran Sales vs Profit")
        st.plotly_chart(fig4, use_container_width=True)

# Tabel ringkas Top 10 dengan margin
st.subheader("Ringkasan 10 Sub-Category Teratas (berdasarkan Sales)")
if {"Sub-Category", "Sales"}.issubset(fdf.columns):
    if "Order ID" in fdf:
        orders_agg = ("Order ID", "nunique")
    else:
        orders_agg = ("Sales", "count")

    tab = (
        fdf.groupby("Sub-Category", as_index=False)
           .agg(Sales=("Sales","sum"),
                Profit=("Profit","sum"),
                Orders=orders_agg)
           .sort_values("Sales", ascending=False)
           .head(10)
    )
    tab["Margin%"] = (tab["Profit"] / tab["Sales"]).replace([float("inf"), -float("inf")], 0.0).fillna(0.0) * 100

    st.dataframe(tab[["Sub-Category", "Sales", "Profit", "Margin%", "Orders"]],
                 use_container_width=True, height=360)

    st.download_button("Unduh ringkasan (CSV)",
                       data=tab.to_csv(index=False).encode("utf-8"),
                       file_name="top10_subcategory.csv", mime="text/csv")
else:
    st.info("Kolom Sub-Category atau Sales tidak tersedia.")
