import numpy as np
import pandas as pd

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
ID_COL = "Transaction_ID"
DATE_COL = "Date"
TARGET = "Units_Sold"
NUMERIC_COLS = [
    "Units_Sold", "Unit_Price", "Discount_Percent", "Inventory_Level", "Competitor_Price",
    "Customer_Footfall", "Marketing_Spend", "Economic_Index", "Online_Orders", "Returns",
    "Customer_Satisfaction", "Shipping_Time", "Supplier_Rating", "Sales_Amount",
]
INT_COLS = ["Units_Sold", "Discount_Percent", "Inventory_Level", "Customer_Footfall",
            "Online_Orders", "Returns", "Shipping_Time"]
TEXT_COLS = ["Store_ID", "Store_Type", "Region", "City", "Product_ID", "Product_Category",
             "Product_Subcategory", "Brand", "Day_Of_Week", "Season", "Weather_Condition", "Demand_Level"]
FLAG_COLS = ["Promotion", "Holiday"] # "Yes"/"No" -> 1/0
FLAG_MAP = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
NON_NEGATIVE_COLS = ["Units_Sold", "Unit_Price", "Inventory_Level", "Competitor_Price",
                     "Customer_Footfall", "Marketing_Spend", "Online_Orders", "Returns",
                     "Shipping_Time", "Sales_Amount"]
POSITIVE_COLS = ["Units_Sold", "Unit_Price", "Competitor_Price"]   # a sale or a price of 0 is impossible too (and Price_Gap divides by Competitor_Price)
RATING_COLS = ["Customer_Satisfaction", "Supplier_Rating"] # 1-5 scores

# Outliers: IQR rule
OUTLIER_COLS = ["Units_Sold", "Unit_Price", "Inventory_Level", "Customer_Footfall", "Marketing_Spend"]
OUTLIER_GROUP = "Product_Category"


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def data_clean(df_load, outlier_action="report"):

    if outlier_action not in ("report", "cap"):
        raise ValueError("outlier_action must be 'report' or 'cap'")
    df_clean = df_load.copy()
    rows_in = len(df_clean)

    # 1. Check for inconsistent data types
    print(LINE)
    print("1. Data types")
    for col in NUMERIC_COLS:
        converted = pd.to_numeric(df_clean[col], errors="coerce")
        n_bad = int((converted.isna() & df_clean[col].notna()).sum())
        if n_bad:
            print(f"{col:<22}: {n_bad:,} unreadable entries -> set to missing")
        df_clean[col] = converted
    dates = pd.to_datetime(df_clean[DATE_COL], errors="coerce", format="mixed")
    n_bad = int((dates.isna() & df_clean[DATE_COL].notna()).sum())
    if n_bad:
        print(f"{DATE_COL:<22}: {n_bad:,} unreadable dates -> set to missing")
    df_clean[DATE_COL] = dates
    print("Numeric columns converted to numbers, Date converted to a date type.")
    print(LINE + "\n")

    # 2. Check for inconsistent data formats
    print(LINE)
    print("2. Data formats")
    n_fixed = 0
    for col in TEXT_COLS:
        text = df_clean[col].str.strip().str.replace(r"\s+", " ", regex=True)       # extra spaces
        known = text.dropna()
        spelling = known.groupby(known.str.lower()).agg(lambda s: s.mode().iloc[0])   # most common spelling
        tidy = text.str.lower().map(spelling)                                  # "sunny" / "SUNNY" -> "Sunny"
        n_fixed += int((tidy.notna() & (tidy != df_clean[col])).sum())
        df_clean[col] = tidy
    print(f"Text columns: {n_fixed:,} values tidied (extra spaces, upper/lower case).")
    for col in FLAG_COLS:
        flags = df_clean[col].astype(str).str.strip().str.lower().map(FLAG_MAP)
        n_bad = int((flags.isna() & df_clean[col].notna()).sum())
        if n_bad:
            print(f"{col:<22}: {n_bad:,} entries not Yes/No -> set to missing")
        df_clean[col] = flags
    print("Promotion and Holiday: Yes/No turned into 1/0.")
    print(LINE + "\n")

    # 3. Check for missing values
    print(LINE)
    print("3. Missing values")
    missing = df_clean.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(pd.DataFrame({"missing": missing, "percent": (missing / len(df_clean) * 100).round(2)}).to_string())
        # Without a date or the target the row cannot be used, so it goes; the rest are filled
        n_before = len(df_clean)
        df_clean = df_clean.dropna(subset=[DATE_COL, TARGET]).reset_index(drop=True)
        if len(df_clean) < n_before:
            print(f"Dropped {n_before - len(df_clean):,} rows with a missing {DATE_COL} or {TARGET}.")
        # NOTE: medians/modes here come from the whole file, before the train/test split. The current raw
        # data has no missing cells, so this never fires; if it starts to, fit the fill values on the
        # training rows only (e.g. an imputer inside the ColumnTransformer in data_modeling).
        for col in NUMERIC_COLS:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in TEXT_COLS + FLAG_COLS:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
        print("Filled: numbers with the median, text and Yes/No columns with the most common value.")
    print(LINE + "\n")

    # 4. Check for duplicate rows
    print(LINE)
    print("4. Duplicate rows")
    same_content = [c for c in df_clean.columns if c != ID_COL]      # a repeated transaction gets a new ID
    n_dup = int(df_clean.duplicated(subset=same_content).sum())
    if n_dup == 0:
        print("No duplicate rows found.")
    else:
        df_clean = df_clean.drop_duplicates(subset=same_content).reset_index(drop=True)
        print(f"Dropped {n_dup:,} duplicate rows.")
    # Repeated Date + Store + Product keys are reported, not dropped: in this dataset the IDs are not stable
    # (one Product_ID spans several categories and prices), so such rows are different sales, not copies.
    key = ["Date", "Store_ID", "Product_ID"]
    n_key = int(df_clean.duplicated(subset=key, keep=False).sum())
    if n_key:
        n_same = int(df_clean.duplicated(subset=key + ["Product_Category", "Unit_Price"], keep=False).sum())
        print(f"{n_key:,} rows share a Date-Store-Product key with another row; {n_same:,} of them also share "
              f"category and price (kept - only exact copies are dropped).")
    print(LINE + "\n")

    # 5. Check for invalid values (impossible, not just unusual)
    print(LINE)
    print("5. Invalid values")
    checks = {f"{c} below 0": (df_clean[c] < 0, c) for c in NON_NEGATIVE_COLS if c not in POSITIVE_COLS}
    checks.update({f"{c} 0 or below": (df_clean[c] <= 0, c) for c in POSITIVE_COLS})
    checks["Discount outside 0-100"] = (~df_clean["Discount_Percent"].between(0, 100), "Discount_Percent")
    for c in RATING_COLS:
        checks[f"{c} outside 1-5"] = (~df_clean[c].between(1, 5), c)
    n_found = 0
    for label, (mask, col) in checks.items():
        if mask.any():
            df_clean.loc[mask, col] = df_clean.loc[~mask, col].median()
            print(f"{label:<28}: {int(mask.sum()):,} rows -> replaced with the median")
            n_found += 1
    if n_found == 0:
        print("No negative values, impossible discounts or bad ratings found.")

    # Consistency checks: reported only, nothing is changed
    promo_no_disc = int(((df_clean["Promotion"] == 0) & (df_clean["Discount_Percent"] > 0)).sum())
    promo_yes_nodisc = int(((df_clean["Promotion"] == 1) & (df_clean["Discount_Percent"] == 0)).sum())
    expected = df_clean["Units_Sold"] * df_clean["Unit_Price"] * (1 - df_clean["Discount_Percent"] / 100)
    n_amount = int(((df_clean["Sales_Amount"] - expected).abs() > 1.0).sum())
    print("Consistency (reported, not changed):")
    print(f"  Promotion = No but Discount > 0      : {promo_no_disc:,} rows")
    print(f"  Promotion = Yes but Discount = 0     : {promo_yes_nodisc:,} rows")
    print(f"  Sales_Amount != Units x Price x (1-Discount) by more than 1 : {n_amount:,} rows")
    print(LINE + "\n")

    # 6. Check for outliers (IQR rule per product category)
    print(LINE)
    print("6. Outliers")
    print(f"Rule   : below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR, judged within each {OUTLIER_GROUP}")
    print(f"Action : {outlier_action}")
    table = []
    for col in OUTLIER_COLS:
        group = df_clean.groupby(OUTLIER_GROUP)[col]
        q1 = group.transform(lambda s: s.quantile(0.25))
        q3 = group.transform(lambda s: s.quantile(0.75))
        lower = (q1 - 1.5 * (q3 - q1)).clip(lower=0)
        upper = q3 + 1.5 * (q3 - q1)
        mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        treated = "left as is"
        if outlier_action == "cap" and col != TARGET and mask.any():
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            treated = "capped"
        table.append({"column": col, "outliers": int(mask.sum()),
                      "percent": round(mask.mean() * 100, 2), "action": treated})
    print(pd.DataFrame(table).to_string(index=False))
    print("Big sales days are real demand (promotions and holidays), so they are kept by default.")
    print(LINE + "\n")

    # Whole-number columns go back to whole numbers
    df_clean[INT_COLS] = df_clean[INT_COLS].round().astype(int)
    df_clean[FLAG_COLS] = df_clean[FLAG_COLS].astype(int)

    # 7. Summary
    print(LINE)
    print("7. Cleaning summary")
    print(f"Rows in  : {rows_in:,}")
    print(f"Rows out : {len(df_clean):,}  ({rows_in - len(df_clean):,} removed)")
    print(f"Missing cells left : {int(df_clean.isna().sum().sum()):,}")
    print(LINE + "\n")

    return df_clean