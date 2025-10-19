import pandas as pd
import numpy as np

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for ML & EDA.
    - Days_to_Ship
    - Year, Month (from Order Date)
    - Profitable (target) = Profit > 0
    - Clean/cap some odd values
    """
    out = df.copy()

    # Days_to_Ship
    if "Order Date" in out.columns and "Ship Date" in out.columns:
        out["Days_to_Ship"] = (out["Ship Date"] - out["Order Date"]).dt.days

    # Calendar features
    if "Order Date" in out.columns:
        out["Order_Year"] = out["Order Date"].dt.year
        out["Order_Month"] = out["Order Date"].dt.to_period("M").astype(str)

    # Target
    if "Profit" in out.columns:
        out["Profitable"] = (out["Profit"] > 0).astype("Int64")

    # Basic sanity caps
    if "Discount" in out.columns:
        out["Discount"] = out["Discount"].clip(lower=0, upper=0.9)
    if "Quantity" in out.columns:
        out["Quantity"] = out["Quantity"].clip(lower=1)
    if "Days_to_Ship" in out.columns:
        out.loc[out["Days_to_Ship"] < -1, "Days_to_Ship"] = np.nan  # invalid negatives

    return out
