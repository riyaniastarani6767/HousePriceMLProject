import streamlit as st
import pandas as pd
# --- add project root to sys.path ---
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # => folder project root
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ------------------------------------

from src.visualization import (
    kpi_summary, bar_sales_profit_by_category, top_subcategory_by_sales,
    monthly_trend, scatter_sales_profit
)

st.set_page_config(page_title="EDA Analysis", page_icon="📊", layout="wide")
st.title("📊 EDA – Business Overview")

df = st.session_state.get("data_df", pd.DataFrame())
if df.empty:
    st.warning("Tidak ada data. Kembali ke Home untuk upload/gunakan default.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("🔎 Filters")
    region = st.multiselect("Region", sorted(df["Region"].dropna().unique().tolist()) if "Region" in df else [])
    category = st.multiselect("Category", sorted(df["Category"].dropna().unique().tolist()) if "Category" in df else [])
    subcat_all = sorted(df["Sub-Category"].dropna().unique().tolist()) if "Sub-Category" in df else []
    # Cascade
    if category and "Category" in df and "Sub-Category" in df:
        subcat_all = sorted(df[df["Category"].isin(category)]["Sub-Category"].dropna().unique().tolist())
    subcat = st.multiselect("Sub-Category", subcat_all)

    date_range = None
    if "Order Date" in df.columns:
        min_d = df["Order Date"].min()
        max_d = df["Order Date"].max()
        date_range = st.date_input("Order Date Range", (min_d, max_d))

# Apply filters
mask = pd.Series([True]*len(df))
if region and "Region" in df: mask &= df["Region"].isin(region)
if category and "Category" in df: mask &= df["Category"].isin(category)
if subcat and "Sub-Category" in df: mask &= df["Sub-Category"].isin(subcat)
if date_range and "Order Date" in df.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask &= df["Order Date"].between(start, end)

fdf = df[mask].copy()

# KPIs
kpi = kpi_summary(fdf)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"${kpi['total_sales']:,.2f}")
c2.metric("Total Profit", f"${kpi['total_profit']:,.2f}")
c3.metric("Unique Orders", f"{kpi['n_orders']:,}")
c4.metric("Profit Ratio", f"{kpi['profit_ratio']:.2f}%" if kpi['profit_ratio'] is not None else "—")

# Charts
fig1 = bar_sales_profit_by_category(fdf)
fig2 = top_subcategory_by_sales(fdf, top_n=10)
fig3 = monthly_trend(fdf)
fig4 = scatter_sales_profit(fdf)

for fig in [fig1, fig2, fig3, fig4]:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
