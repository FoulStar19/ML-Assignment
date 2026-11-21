import numpy as np
import pandas as pd

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS  (raw column names, i.e. before data_transforming renames them)
# ─────────────────────────────────────────────
ID_COL = "Transaction_ID"                           # one row = one transaction
KEY_COLS = ["Date", "Store_ID", "Product_ID"]       # a row without these cannot be placed
GROUP_COL = "Product_Subcategory"                   # stable grouping used to fill gaps and to judge outliers
NUMERIC_COLS = [
    "Units_Sold", "Unit_Price", "Discount_Percent", "Inventory_Level", "Competitor_Price",
    "Customer_Footfall", "Marketing_Spend", "Economic_Index", "Online_Orders", "Returns",
    "Customer_Satisfaction", "Shipping_Time", "Supplier_Rating", "Sales_Amount",
]
NON_NEGATIVE_COLS = [
    "Units_Sold", "Unit_Price", "Inventory_Level", "Competitor_Price", "Customer_Footfall",
    "Marketing_Spend", "Online_Orders", "Returns", "Shipping_Time", "Sales_Amount",
]
RATING_COLS = ["Customer_Satisfaction", "Supplier_Rating"]    # both are 1-5 scores
FLAG_COLS = ["Promotion", "Holiday"]                          # "Yes"/"No" in the raw file -> 1/0
FLAG_MAP = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}

# Allowed values for the small categorical columns (anything else is reported)
CATEGORY_SUBCATEGORIES = {
    "Beauty": {"Fragrance", "Haircare", "Makeup", "Skincare"},
    "Clothing": {"Footwear", "Kids", "Men", "Women"},
    "Electronics": {"Accessories", "Audio", "Laptop", "Mobile"},
    "Groceries": {"Beverages", "Dairy", "Fresh Produce", "Snacks"},
    "Home & Kitchen": {"Cookware", "Decor", "Furniture", "Storage"},
    "Sports": {"Fitness", "Outdoor", "Sportswear", "Team Sports"},
    "Toys": {"Action Figures", "Board Games", "Educational", "Puzzles"},
}
VALID_VALUES = {
    "Store_Type": {"Supermarket", "Hypermarket", "Convenience Store", "Online Store"},
    "Region": {"North", "South", "East", "West", "Central"},
    "City": {"Atlanta", "Boston", "Chicago", "Dallas", "Denver", "Houston", "Los Angeles",
             "New York", "Phoenix", "Seattle"},
    "Product_Category": set(CATEGORY_SUBCATEGORIES),
    "Product_Subcategory": set().union(*CATEGORY_SUBCATEGORIES.values()),
    "Brand": {"Apex", "EcoLife", "FreshMart", "Nova", "PrimePlus", "UrbanCo", "ValueMax", "Zenith"},
    "Weather_Condition": {"Sunny", "Rainy", "Cloudy", "Snowy", "Stormy"},
    "Season": {"Autumn", "Winter", "Spring", "Summer"},
    "Demand_Level": {"Low", "Medium", "High", "Very High"},
}
ID_PATTERNS = {"Store_ID": r"^ST\d{3}$", "Product_ID": r"^PR\d{4}$"}
TEXT_COLS = list(VALID_VALUES) + list(ID_PATTERNS) + ["Day_Of_Week"]

# Outlier check: IQR rule, judged inside each subcategory (a laptop price is not judged against a snack price).
# Discount (fixed steps), Returns (mostly 0/1), the 1-5 ratings and the economic index are skipped.
OUTLIER_COLS = ["Units_Sold", "Unit_Price", "Inventory_Level", "Competitor_Price",
                "Customer_Footfall", "Marketing_Spend", "Online_Orders", "Sales_Amount"]
OUTLIER_GROUP = GROUP_COL
IQR_K = 1.5                 # fences = Q1 - 1.5*IQR  and  Q3 + 1.5*IQR
MAX_OUTLIER_SHARE = 0.05    # a column with more outliers than this is skewed, not "outlier-y" -> never capped/removed
SALES_TOLERANCE = 0.05      # Sales_Amount may differ this much from the recomputed value (rounding)


def _fill_missing(df, col):
    # Fill only the empty cells: median of the same subcategory, then of the whole column
    by_group = df.groupby(GROUP_COL)[col].transform("median")
    return df[col].fillna(by_group).fillna(df[col].median())


def _to_flag(series):
    # "Yes"/"No" (any case, spacing) -> 1/0; anything else -> NaN
    return series.astype(str).str.strip().str.lower().map(FLAG_MAP).astype(float)


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def data_clean(df_load, outlier_action="report"):
    """Check and clean the raw data.

    outlier_action: "report" (default) only lists the outliers and leaves the data alone,
                    "flag" adds a 0/1 <column>_is_outlier column,
                    "cap" clips them to the IQR fences, "remove" drops those rows.
    """
    df_clean = df_load.copy()
    rows_in = len(df_clean)

    absent = [c for c in [ID_COL] + KEY_COLS + NUMERIC_COLS + FLAG_COLS + list(VALID_VALUES) if c not in df_clean.columns]
    if absent:
        raise ValueError(f"Raw data is missing expected columns: {absent}")

    # Tidy text first so "ST001 " and "ST001" are not counted as different stores, turn Yes/No into 1/0,
    # and force numeric columns to numbers (unreadable values become missing and are handled in step 1)
    for col in TEXT_COLS:
        if col in df_clean.columns and pd.api.types.is_string_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].str.strip()
    unreadable = {}
    for col in FLAG_COLS:
        converted = _to_flag(df_clean[col])
        n_bad = int((converted.isna() & df_clean[col].notna()).sum())
        if n_bad:
            unreadable[col] = n_bad
        df_clean[col] = converted
    for col in NUMERIC_COLS:
        converted = pd.to_numeric(df_clean[col], errors="coerce")
        n_bad = int((converted.isna() & df_clean[col].notna()).sum())
        if n_bad:
            unreadable[col] = n_bad
        df_clean[col] = converted

    # 1. Check for Missing Values
    print(LINE)
    print("1. Missing values")
    if unreadable:
        print(f"Unreadable entries turned into missing values: {unreadable}")
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
            print(f"Dropped {n_before - len(df_clean):,} rows with a missing Date / Store_ID / Product_ID.")
        for col in VALID_VALUES:                       # text first, so the subcategory is complete for the numbers
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
        for col in NUMERIC_COLS:
            if df_clean[col].isna().any():
                df_clean[col] = _fill_missing(df_clean, col)
        for col in FLAG_COLS:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
        print("Filled: numbers with the subcategory median, flags and text with the most common value.")
    print(LINE + "\n")

    # 2. Check for Duplicate Rows
    print(LINE)
    print("2. Duplicate rows")
    # The same transaction logged twice gets a new Transaction_ID, so look at every column except the ID
    same_content = [c for c in df_clean.columns if c != ID_COL]
    n_dup = int(df_clean.duplicated(subset=same_content).sum())
    if n_dup == 0:
        print("No duplicate rows found.")
    else:
        print(f"Duplicate rows : {n_dup:,} ({n_dup / len(df_clean) * 100:.2f}%)")
        print("First few duplicates:")
        print(df_clean[df_clean.duplicated(subset=same_content, keep="first")].head().to_string())
        df_clean = df_clean.drop_duplicates(subset=same_content).reset_index(drop=True)
        print(f"Dropped {n_dup:,} duplicate rows.")
    n_id_dup = int(df_clean[ID_COL].duplicated().sum())
    print(f"{ID_COL} is unique." if n_id_dup == 0 else
          f"WARNING: {n_id_dup:,} rows repeat a {ID_COL} (not dropped, please check).")
    # Same day + store + product twice is possible (two separate transactions), so it is only reported
    n_key_dup = int(df_clean.duplicated(subset=KEY_COLS).sum())
    print("Every Date + Store_ID + Product_ID is unique." if n_key_dup == 0 else
          f"{n_key_dup:,} rows share a Date + Store_ID + Product_ID with another row (separate transactions, kept).")
    print(LINE + "\n")

    # 3. Check text values, IDs, dates and calendar columns
    print(LINE)
    print("3. Text values, dates and calendar columns")
    for col, allowed in VALID_VALUES.items():
        found = set(df_clean[col].dropna().unique())
        unexpected = sorted(found - allowed)
        if unexpected:
            print(f"{col:<20}: unexpected values {unexpected}")
        else:
            print(f"{col:<20}: OK ({len(found)} values)")
    n_mismatch = int((~df_clean.apply(
        lambda r: r["Product_Subcategory"] in CATEGORY_SUBCATEGORIES.get(r["Product_Category"], set()), axis=1)).sum())
    print(f"{'Subcategory':<20}: " + ("every subcategory sits under its own category." if n_mismatch == 0
                                      else f"{n_mismatch:,} rows have a subcategory that does not belong to the category."))
    for col, pattern in ID_PATTERNS.items():
        n_odd = int((~df_clean[col].astype(str).str.match(pattern)).sum())
        print(f"{col:<20}: {df_clean[col].nunique()} unique, " +
              ("all match the usual format." if n_odd == 0 else f"{n_odd:,} rows do not match the usual format."))
    dates = pd.to_datetime(df_clean["Date"], errors="coerce")
    n_bad_dates = int(dates.isna().sum())
    print(f"{'Date':<20}: {n_bad_dates:,} unreadable dates (data_transforming drops them)"
          if n_bad_dates else f"{'Date':<20}: OK ({dates.min().date()} → {dates.max().date()})")
    expected = {"Year": dates.dt.year, "Month": dates.dt.month, "Day": dates.dt.day,
                "Quarter": dates.dt.quarter, "Week": dates.dt.isocalendar().week, "Day_Of_Week": dates.dt.day_name()}
    off = {c: int((dates.notna() & (df_clean[c].astype(str) != e.astype(str))).sum())
           for c, e in expected.items() if c in df_clean.columns}
    off = {c: n for c, n in off.items() if n}
    print(f"{'Calendar columns':<20}: " + ("all agree with Date." if not off else
          f"disagree with Date: {off} (data_transforming recomputes them from Date)."))
    season_of_month = df_clean.groupby(dates.dt.month)["Season"].agg(lambda s: s.mode().iloc[0])
    n_season = int((dates.notna() & (dates.dt.month.map(season_of_month) != df_clean["Season"])).sum())
    print(f"{'Season':<20}: " + ("every month has one season." if n_season == 0 else
                                 f"{n_season:,} rows differ from the usual season of their month."))
    print(LINE + "\n")

    # 4. Check for invalid values (impossible, not just unusual)
    print(LINE)
    print("4. Invalid values")
    problems = []   # (label, column, mask)
    for col in NON_NEGATIVE_COLS:
        mask = df_clean[col] < 0
        if mask.any():
            problems.append((f"{col} < 0", col, mask))
    mask = ~df_clean["Discount_Percent"].between(0, 100)
    if mask.any():
        problems.append(("Discount outside 0-100", "Discount_Percent", mask))
    for col in RATING_COLS:
        mask = ~df_clean[col].between(1, 5)
        if mask.any():
            problems.append((f"{col} outside 1-5", col, mask))
    mask = df_clean["Returns"] > df_clean["Units_Sold"]
    if mask.any():
        problems.append(("Returns > Units_Sold", "Returns", mask))
    if not problems:
        print("No negative values, impossible discounts, bad ratings or returns above sales found.")
    for label, col, mask in problems:
        if label == "Returns > Units_Sold":
            df_clean.loc[mask, "Returns"] = df_clean.loc[mask, "Units_Sold"]
            print(f"{label:<28}: {int(mask.sum()):,} rows -> Returns set to Units_Sold")
        else:
            df_clean.loc[mask, col] = np.nan
            df_clean[col] = _fill_missing(df_clean, col)
            print(f"{label:<28}: {int(mask.sum()):,} rows -> replaced with the subcategory median")
    # Odd-looking but possibly real, so these are only reported
    n_over = int((df_clean["Units_Sold"] > df_clean["Inventory_Level"]).sum())
    print(f"Units_Sold > Inventory_Level : {n_over:,} rows (reported only)")
    n_zero = int((df_clean["Sales_Amount"] == 0).sum())
    print(f"Sales_Amount = 0             : {n_zero:,} rows (reported only)")
    recomputed = (df_clean["Units_Sold"] - df_clean["Returns"]) * df_clean["Unit_Price"] * (1 - df_clean["Discount_Percent"] / 100)
    n_recon = int(((recomputed - df_clean["Sales_Amount"]).abs() > SALES_TOLERANCE).sum())
    print("Sales_Amount = (Units_Sold - Returns) x Unit_Price x (1 - Discount%) on every row." if n_recon == 0 else
          f"Sales_Amount differs from (Units_Sold - Returns) x Unit_Price x (1 - Discount%) on {n_recon:,} rows (reported only).")
    print(LINE + "\n")

    # 5. Check the structure of the data (what history each store / product has)
    # Rows are single transactions, not one row per store-product per day, so this decides what the
    # later steps may assume.
    print(LINE)
    print("5. Data structure")
    d = pd.to_datetime(df_clean["Date"], errors="coerce")
    per_day = d.value_counts()
    span = (d.max() - d.min()).days + 1
    print(f"Days with data  : {d.nunique():,} of {span:,} ({per_day.min()}-{per_day.max()} rows a day, median {int(per_day.median())})")
    per_line = df_clean.groupby(["Store_ID", "Product_ID"]).size()
    single = float((per_line == 1).mean())
    print(f"Store-product lines : {len(per_line):,}, {per_line.mean():.2f} rows each on average, {single:.0%} have a single row")
    prod_cats = df_clean.groupby("Product_ID")["Product_Category"].nunique()
    store_types = df_clean.groupby("Store_ID")["Store_Type"].nunique()
    print(f"Product_ID          : {int((prod_cats > 1).sum())} of {len(prod_cats)} IDs appear under more than one category "
          f"(on average {prod_cats.mean():.1f})")
    print(f"Store_ID            : {int((store_types > 1).sum())} of {len(store_types)} IDs appear under more than one store type "
          f"(on average {store_types.mean():.1f})")
    if per_line.median() < 7 or (prod_cats > 1).mean() > 0.5:
        print("-> Store and product IDs are labels, not stable entities, and lines have no daily history. Every row is\n"
              "   therefore treated as one observation (no per-line lags or rolling windows are built).")
    print(LINE + "\n")

    # 6. Check for Outliers (IQR rule per subcategory)
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
        col, _, grp = worst
        s = df_clean.loc[df_clean[OUTLIER_GROUP] == grp, col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        print(f"\nWorked example ({grp}, {col}): Q1 = {q1:g}, Q3 = {q3:g}, IQR = {q3 - q1:g}")
        print(f"  lower fence = {q1:g} - {IQR_K:g} x {q3 - q1:g} = {q1 - IQR_K * (q3 - q1):g}")
        print(f"  upper fence = {q3:g} + {IQR_K:g} x {q3 - q1:g} = {q3 + IQR_K * (q3 - q1):g}")
        print("Outliers are kept unless they are clearly errors: big sales days are real demand, and "
              "Units_Sold is what the stock-status label and the demand model learn from.")
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
