import pandas as pd

def read_superstore_csv(path: str) -> pd.DataFrame:
    """
    Read Superstore CSV that uses semicolon (;) separator and comma decimal (e.g., 261,96).
    Auto-fixes common types: dates, numeric columns, strip spaces.
    """
    # Read as string first to normalize manually
    df = pd.read_csv(path, sep=';', dtype=str, encoding='latin1')

    # Standardize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Expected numeric columns (present in many Superstore variants)
    num_cols = ["Sales", "Profit", "Quantity", "Discount"]
    for col in num_cols:
        if col in df.columns:
            # Replace decimal comma with dot, remove thousand separators if any
            df[col] = (
                df[col]
                .str.replace('.', '', regex=False)  # remove thousands like 1.234,56 (EU)
                .str.replace(',', '.', regex=False) # decimal comma -> dot
            )
            # Quantity should be int if possible
            if col == "Quantity":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Dates
    for dcol in ["Order Date", "Ship Date", "Order Date ", "Ship Date "]:
        if dcol in df.columns:
            df[dcol.strip()] = pd.to_datetime(df[dcol], errors='coerce', dayfirst=True)

    # Trim whitespace for categoricals
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)

    # Unify canonical date column names
    if "Order Date " in df.columns: df = df.rename(columns={"Order Date ": "Order Date"})
    if "Ship Date " in df.columns:  df = df.rename(columns={"Ship Date ": "Ship Date"})

    return df
