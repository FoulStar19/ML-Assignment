import numpy as np
import pandas as pd

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS  (raw column names, i.e. before data_transforming renames them)
# ─────────────────────────────────────────────
KEY_COLS = ["Date", "Store ID", "Product ID"]       # one row = one product, in one store, on one day
LINE_KEYS = ["Store ID", "Product ID"]
ID_COLS = ["Store ID", "Product ID"]
NUMERIC_COLS = ["Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
                "Price", "Discount", "Competitor Pricing"]
NON_NEGATIVE_COLS = NUMERIC_COLS
FLOOR_AT_ZERO_COLS = ["Demand Forecast"]   # a forecast below 0 is noise around a tiny demand, so 0 is its best fix
BINARY_COLS = ["Holiday/Promotion"]
# Allowed values for the small categorical columns (anything else is reported)
VALID_VALUES = {
    "Category": {"Groceries", "Toys", "Electronics", "Furniture", "Clothing"},
    "Region": {"North", "South", "East", "West"},
    "Weather Condition": {"Sunny", "Rainy", "Cloudy", "Snowy"},
    "Seasonality": {"Autumn", "Winter", "Spring", "Summer"},
}
# Outlier check: IQR rule, judged inside each product (a hit for a toy is not judged against a sofa)
OUTLIER_COLS = ["Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
                "Price", "Competitor Pricing"]   # Discount is a fixed set of values (0/5/10/15/20), so it is skipped
OUTLIER_GROUP = "Product ID"
IQR_K = 1.5                 # fences = Q1 - 1.5*IQR  and  Q3 + 1.5*IQR
MAX_OUTLIER_SHARE = 0.05    # a column with more outliers than this is skewed, not "outlier-y" -> never capped/removed


def _fill_missing(df, col):
    # Fill only the empty cells: median of the same store-product line, then of the product, then of the column
    by_line = df.groupby(LINE_KEYS)[col].transform("median")
    by_product = df.groupby("Product ID")[col].transform("median")
    return df[col].fillna(by_line).fillna(by_product).fillna(df[col].median())


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def data_clean(df_load, outlier_action="report"):
    """Check and clean the raw data.

    outlier_action: "report" (default) only lists the outliers and leaves the data alone,
                    "flag" adds a 0/1 <column>_is_outlier column,
                    "cap" clips them to the IQR fences, "remove" drops those rows
                    (not recommended: a dropped day breaks the one-row-per-day rolling features).
    """
    df_clean = df_load.copy()
    rows_in = len(df_clean)

    absent = [c for c in KEY_COLS + NUMERIC_COLS + list(VALID_VALUES) if c not in df_clean.columns]
    if absent:
        raise ValueError(f"Raw data is missing expected columns: {absent}")

    # Tidy text first so "s001 " and "S001" are not counted as different stores,
    # and force numeric columns to numbers (unreadable values become missing and are handled in step 1)
    for col in list(VALID_VALUES) + ID_COLS:
        if pd.api.types.is_string_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].str.strip()
    unreadable = {}
    for col in NUMERIC_COLS + BINARY_COLS:
        if col in df_clean.columns:
            converted = pd.to_numeric(df_clean[col], errors="coerce")
            n_bad = int((converted.isna() & df_clean[col].notna()).sum())
            if n_bad:
                unreadable[col] = n_bad
            df_clean[col] = converted

    # 1. Check for Missing Values
    print(LINE)
    print("1. Missing values")
    if unreadable:
        print(f"Non-numeric entries turned into missing values: {unreadable}")
    missing = df_clean.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        report = pd.DataFrame({
            "missing": missing,
            "percent": (missing / len(df_clean) * 100).round(2),
        })
        print(report.to_string())
        print(f"\nTotal missing cells : {int(missing.sum()):,}")
        print(f"Rows with any missing : {int(df_clean.isna().any(axis=1).sum()):,}")

        # Fix: rows without a key cannot be placed, so they go; the rest are filled
        n_before = len(df_clean)
        df_clean = df_clean.dropna(subset=KEY_COLS).reset_index(drop=True)
        if len(df_clean) < n_before:
            print(f"Dropped {n_before - len(df_clean):,} rows with a missing Date / Store ID / Product ID.")
        for col in NUMERIC_COLS:
            if df_clean[col].isna().any():
                df_clean[col] = _fill_missing(df_clean, col)
        for col in BINARY_COLS:
            df_clean[col] = df_clean[col].fillna(0)
        for col in list(VALID_VALUES):
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
        print("Filled: numbers with the store-product median, flags with 0, text with the most common value.")
    print(LINE + "\n")

    # 2. Check for Duplicate Rows
    print(LINE)
    print("2. Duplicate rows")
    n_dup = int(df_clean.duplicated().sum())
    if n_dup == 0:
        print("No duplicate rows found.")
    else:
        print(f"Duplicate rows : {n_dup:,} ({n_dup / len(df_clean) * 100:.2f}%)")
        print("First few duplicates:")
        print(df_clean[df_clean.duplicated(keep="first")].head().to_string())
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"Dropped {n_dup:,} duplicate rows.")
    # Same day + store + product twice with different numbers would break the rolling features
    n_key_dup = int(df_clean.duplicated(subset=KEY_COLS).sum())
    if n_key_dup == 0:
        print("Every Date + Store ID + Product ID is unique.")
    else:
        print(f"WARNING: {n_key_dup:,} rows repeat a Date + Store ID + Product ID (not dropped, please check).")
    print(LINE + "\n")

    # 3. Check text values and dates
    print(LINE)
    print("3. Text values and dates")
    for col, allowed in VALID_VALUES.items():
        found = set(df_clean[col].dropna().unique())
        unexpected = sorted(found - allowed)
        if unexpected:
            print(f"{col:<18}: unexpected values {unexpected}")
        else:
            print(f"{col:<18}: OK ({len(found)} values)")
    for col in ID_COLS:
        pattern = r"^S\d{3}$" if col == "Store ID" else r"^P\d{4}$"
        n_odd = int((~df_clean[col].astype(str).str.match(pattern)).sum())
        print(f"{col:<18}: {df_clean[col].nunique()} unique, " +
              ("all match the usual format." if n_odd == 0 else f"{n_odd:,} rows do not match the usual format."))
    dates = pd.to_datetime(df_clean["Date"], errors="coerce")
    n_bad_dates = int(dates.isna().sum())
    print(f"{'Date':<18}: {n_bad_dates:,} unreadable dates (data_transforming drops them)"
          if n_bad_dates else f"{'Date':<18}: OK ({dates.min().date()} → {dates.max().date()})")
    print(LINE + "\n")

    # 4. Check for invalid values (impossible, not just unusual)
    print(LINE)
    print("4. Invalid values")
    problems = []   # (label, column, mask)
    for col in NON_NEGATIVE_COLS:
        mask = df_clean[col] < 0
        if mask.any():
            problems.append((f"{col} < 0", col, mask))
    mask = ~df_clean["Discount"].between(0, 100)
    if mask.any():
        problems.append(("Discount outside 0-100", "Discount", mask))
    for col in BINARY_COLS:
        mask = ~df_clean[col].isin([0, 1])
        if mask.any():
            problems.append((f"{col} not 0/1", col, mask))
    if not problems:
        print("No negative values, impossible discounts or bad flags found.")
    for label, col, mask in problems:
        if label.endswith("< 0") and col in FLOOR_AT_ZERO_COLS:
            lowest = df_clean.loc[mask, col].min()
            avg_sold = df_clean.loc[mask, "Units Sold"].mean()
            df_clean.loc[mask, col] = 0
            print(f"{label:<28}: {int(mask.sum()):,} rows (lowest {lowest:g}) -> set to 0 "
                  f"(units sold on these rows average only {avg_sold:.1f})")
        else:
            df_clean.loc[mask, col] = np.nan
            df_clean[col] = _fill_missing(df_clean, col)
            print(f"{label:<28}: {int(mask.sum()):,} rows -> replaced with the store-product median")
    # Looks odd but can be real (sold more than what was left on the shelf), so it is only reported
    n_over = int((df_clean["Units Sold"] > df_clean["Inventory Level"]).sum())
    print(f"Units Sold > Inventory Level : {n_over:,} rows (reported only)")
    print(LINE + "\n")

    # 5. Check the daily sequence of every store-product line
    # The history features (lags, 7-day windows) and the stock-status label assume one row per day.
    print(LINE)
    print("5. Daily continuity")
    d = pd.to_datetime(df_clean["Date"], errors="coerce")
    per_line = df_clean.assign(_d=d).groupby(LINE_KEYS)["_d"].agg(["nunique", "min", "max"])
    expected_days = (per_line["max"] - per_line["min"]).dt.days + 1
    gaps = int((expected_days - per_line["nunique"]).clip(lower=0).sum())
    if gaps == 0:
        print(f"All {len(per_line):,} store-product lines have one row for every day ({d.nunique():,} days).")
    else:
        n_lines = int((expected_days > per_line["nunique"]).sum())
        print(f"WARNING: {gaps:,} missing days across {n_lines} lines - rolling features will span the gap.")
    print(LINE + "\n")

    # 6. Check for Outliers (IQR rule per product)
    print(LINE)
    print("6. Outliers")
    print(f"Rule   : below Q1 - {IQR_K:g} x IQR or above Q3 + {IQR_K:g} x IQR, judged within each {OUTLIER_GROUP}")
    print(f"Action : {outlier_action}")
    if outlier_action not in ("report", "flag", "cap", "remove"):
        raise ValueError("outlier_action must be 'report', 'flag', 'cap' or 'remove'")
    drop_mask = pd.Series(False, index=df_clean.index)
    table, worst = [], None
    for col in OUTLIER_COLS:
        g = df_clean.groupby(OUTLIER_GROUP)[col]
        q1 = g.transform(lambda s: s.quantile(0.25))
        q3 = g.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        lower = (q1 - IQR_K * iqr).where(iqr > 0, -np.inf)    # iqr == 0: constant column, nothing to flag
        upper = (q3 + IQR_K * iqr).where(iqr > 0, np.inf)
        is_low, is_high = df_clean[col] < lower, df_clean[col] > upper
        mask = is_low | is_high
        n_out = int(mask.sum())
        share = n_out / len(df_clean)
        if n_out == 0:
            treated = "-"
        elif outlier_action == "report" or share > MAX_OUTLIER_SHARE:
            treated = "left as is"
        elif outlier_action == "flag":
            df_clean[f"{col}_is_outlier"] = mask.astype(int)
            treated = "flagged"
        elif outlier_action == "cap":
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            treated = "capped"
        else:
            drop_mask |= mask
            treated = "removed"
        table.append({
            "column": col, "outliers": n_out, "percent": round(share * 100, 2),
            "below": int(is_low.sum()), "above": int(is_high.sum()),
            "lowest flagged": df_clean.loc[mask, col].min() if n_out else np.nan,
            "highest flagged": df_clean.loc[mask, col].max() if n_out else np.nan,
            "action": treated,
        })
        if n_out and (worst is None or n_out > worst[1]):
            worst = (col, n_out, df_clean.loc[mask, OUTLIER_GROUP].value_counts().index[0])
    print(pd.DataFrame(table).to_string(index=False))
    if worst is None:
        print("\nNo outliers found - nothing to treat.")
    else:
        col, _, prod = worst
        s = df_clean.loc[df_clean[OUTLIER_GROUP] == prod, col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        print(f"\nWorked example ({prod}, {col}): Q1 = {q1:g}, Q3 = {q3:g}, IQR = {q3 - q1:g}")
        print(f"  lower fence = {q1:g} - {IQR_K:g} x {q3 - q1:g} = {q1 - IQR_K * (q3 - q1):g}")
        print(f"  upper fence = {q3:g} + {IQR_K:g} x {q3 - q1:g} = {q3 + IQR_K * (q3 - q1):g}")
        print("Outliers are kept unless they are clearly errors: high-sales days are real demand, and "
              "Units Sold is what the stock-status label and the demand model learn from.")
    if drop_mask.any():
        df_clean = df_clean[~drop_mask].reset_index(drop=True)
        print(f"Removed {int(drop_mask.sum()):,} rows containing outliers.")
    print(LINE + "\n")

    # 7. Summary
    print(LINE)
    print("7. Cleaning summary")
    print(f"Rows in  : {rows_in:,}")
    print(f"Rows out : {len(df_clean):,}  ({rows_in - len(df_clean):,} removed)")
    print(f"Missing cells left : {int(df_clean.isna().sum().sum()):,}")
    print(LINE + "\n")

    return df_clean