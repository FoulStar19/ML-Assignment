import numpy as np
import pandas as pd

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
DATE_COL = "Date"
TARGET = "Units_Sold"
FLAG_COLS = ["Promotion", "Holiday"] # already 1/0 after data_clean
NUM_COLS = ["Month", "Unit_Price", "Discount_Percent", "Marketing_Spend", "Customer_Footfall",
            "Log_Effective_Price", "Price_Gap"] # the last two are created in step 2 below
CAT_COLS = ["Product_Category", "Product_Subcategory", "Weather_Condition", "Store_Type"]
STOCK_COLS = ["Transaction_ID", "Store_ID", "Product_ID", "Inventory_Level"] # not model features, used by the recommender system
COMPETITOR_COL = "Competitor_Price" # only used to build Price_Gap, it is not kept


# ─────────────────────────────────────────────
# 3. DATA TRANSFORMING
# ─────────────────────────────────────────────
def data_transform(df_clean):

    df_transform = df_clean.copy()

    # 1. Create Month from Date
    print(LINE)
    print("1. Month from Date")
    df_transform["Month"] = pd.to_datetime(df_transform[DATE_COL]).dt.month
    print("Month created from Date (the other calendar columns are not needed).")
    print(LINE + "\n")

    # 2. Price features
    print(LINE)
    print("2. Price features")
    effective_price = df_transform["Unit_Price"] * (1 - df_transform["Discount_Percent"] / 100)
    df_transform["Log_Effective_Price"] = np.log(effective_price)                      # price after discount, on a log scale
    df_transform["Price_Gap"] = df_transform["Unit_Price"] / df_transform[COMPETITOR_COL]  # >1 = dearer than the competitor
    print("Log_Effective_Price = log(Unit_Price x (1 - Discount/100)); Price_Gap = Unit_Price / Competitor_Price.")
    print(LINE + "\n")

    # 3. Keep only the columns that are needed
    print(LINE)
    print("3. Columns")
    keep = [DATE_COL] + FLAG_COLS + NUM_COLS + CAT_COLS + [TARGET] + STOCK_COLS
    absent = [c for c in keep if c not in df_transform.columns]
    if absent:
        raise ValueError(f"Cleaned data is missing expected columns: {absent}")
    n_dropped = len(df_transform.columns) - len(keep)
    df_transform = df_transform[keep]
    print(f"Kept {len(FLAG_COLS + NUM_COLS + CAT_COLS)} features, the target ({TARGET}), "
          f"{len(STOCK_COLS)} stock columns and {DATE_COL}; dropped {n_dropped} others.")
    print(LINE + "\n")

    # 4. Sort by date
    print(LINE)
    print("4. Order")
    df_transform = df_transform.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    print("Rows sorted by Date.")
    print(LINE + "\n")

    # 5. Summary
    print(LINE)
    print("5. Transform summary")
    print(f"Rows    : {len(df_transform):,}")
    print(f"Columns : {df_transform.shape[1]}")
    print(f"Missing cells : {int(df_transform.isna().sum().sum()):,}")
    print(LINE + "\n")

    return df_transform