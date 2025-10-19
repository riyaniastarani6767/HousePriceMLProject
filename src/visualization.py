import pandas as pd
import plotly.express as px

def kpi_summary(df: pd.DataFrame) -> dict:
    total_sales = float(df["Sales"].sum()) if "Sales" in df else 0.0
    total_profit = float(df["Profit"].sum()) if "Profit" in df else 0.0
    n_orders = df["Order ID"].nunique() if "Order ID" in df else len(df)
    profit_ratio = (df["Profitable"].mean() * 100.0) if "Profitable" in df else None
    return dict(total_sales=total_sales, total_profit=total_profit, n_orders=n_orders, profit_ratio=profit_ratio)

def bar_sales_profit_by_category(df: pd.DataFrame):
    if not {"Category","Sales","Profit"}.issubset(df.columns): return None
    g = df.groupby("Category", as_index=False)[["Sales","Profit"]].sum()
    return px.bar(g, x="Category", y=["Sales","Profit"], barmode="group", title="Sales & Profit per Category")

def top_subcategory_by_sales(df: pd.DataFrame, top_n:int=10):
    if not {"Sub-Category","Sales"}.issubset(df.columns): return None
    g = df.groupby("Sub-Category", as_index=False)["Sales"].sum().nlargest(top_n, "Sales")
    return px.bar(g, x="Sub-Category", y="Sales", title=f"Top {top_n} Sub-Category by Sales")

def monthly_trend(df: pd.DataFrame):
    if "Order Date" not in df.columns or not {"Sales","Profit"}.issubset(df.columns): return None
    ts = df.set_index("Order Date").sort_index().resample("M")[["Sales","Profit"]].sum().reset_index()
    return px.line(ts, x="Order Date", y=["Sales","Profit"], title="Monthly Sales & Profit Trend")

def scatter_sales_profit(df: pd.DataFrame):
    if not {"Sales","Profit"}.issubset(df.columns): return None
    return px.scatter(df, x="Sales", y="Profit", size="Quantity" if "Quantity" in df.columns else None,
                      color="Discount" if "Discount" in df.columns else None,
                      title="Sales vs Profit (color=Discount, size=Quantity)")
