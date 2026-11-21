import numpy as np
import pandas as pd

LINE = "─" * 45

# Settings for the stock-status label. The model scripts and data_modeling import these too.
# The label compares the stock on hand with the demand of the same record:
#     stock_cover = inventory_level / units_sold      (days the stock lasts, one record = one day of sales)
# Understock if the stock lasts less than UNDERSTOCK_COVER_DAYS, Overstock if more than OVERSTOCK_COVER_DAYS.
# With the current data this gives about 18% Understock, 64% Healthy and 18% Overstock.
UNDERSTOCK_COVER_DAYS = 14.0     # two weeks of cover
OVERSTOCK_COVER_DAYS = 35.0      # five weeks of cover
STOCK_STATUS = ["Understock", "Healthy", "Overstock"]

# Raw names (after snake_case) -> the names the dashboard and model scripts already use
RENAME_COLUMNS = {
    "product_category": "category",
    "unit_price": "price",
    "discount_percent": "discount",
    "competitor_price": "competitor_pricing",
    "season": "seasonality",
    "sales_amount": "revenue",          # final amount after discount and returns
}


# ─────────────────────────────────────────────
# 3. DATA TRANSFORMATION
# ─────────────────────────────────────────────
def standardize_columns(df):
    # 1. Make column names lowercase snake_case, e.g. 'Store_ID' -> 'store_id', then use the project's names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    renamed = {old: new for old, new in RENAME_COLUMNS.items() if old in df.columns}
    df.rename(columns=renamed, inplace=True)
    print(LINE)
    print("1. Column names standardized to lowercase snake_case.")
    if renamed:
        print(f"   Renamed to the project's names: {renamed}")
    print(f"   {list(df.columns)}")
    print(LINE + "\n")


def date_conversion(df, date_column):
    # 2. Convert 'date' column to datetime format
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        print(LINE)
        print(f"2. '{date_column}' column converted to datetime format.")
        print(LINE + "\n")
    else:
        print(LINE)
        print(f"2. No '{date_column}' column found for transformation.")
        print(LINE + "\n")

    # 3. Drop rows where the date could not be parsed
    print(LINE)
    if date_column in df.columns:
        bad_dates = int(df[date_column].isna().sum())
        if bad_dates > 0:
            df.dropna(subset=[date_column], inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"3. {bad_dates:,} rows had unparseable dates and were dropped.")
        else:
            print("3. No unparseable dates found.")
        if df.empty:
            raise ValueError(f"No valid dates left in '{date_column}' after parsing.")
        print(f"Range : {df[date_column].min().date()} → {df[date_column].max().date()}")
    else:
        print(f"3. No '{date_column}' column found for checking unparseable dates.")
    print(LINE + "\n")

    # 4. Create date features (recomputed from the date, so they always agree with it)
    print(LINE)
    if date_column in df.columns:
        df['year'] = df[date_column].dt.year
        df['quarter'] = df[date_column].dt.quarter
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['weekday'] = df[date_column].dt.weekday
        df['month_name'] = df[date_column].dt.month_name()
        df['week'] = df[date_column].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df[date_column].dt.day_name()
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        print("4. Date features created: year, quarter, month, day, weekday, month_name, week, day_of_week, is_weekend.")
    else:
        print(f"4. No '{date_column}' column found for creating date features.")
    print(LINE + "\n")


def numeric_conversion(df, numeric_columns):
    # 5. Convert numeric columns to numeric dtype
    print(LINE)
    converted = []
    missing_cols = []
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted.append(col)
        else:
            missing_cols.append(col)

    if converted:
        print(f"5. Converted to numeric: {converted}")
    if missing_cols:
        print(f"5. Columns not found for numeric conversion: {missing_cols}")
    print(LINE + "\n")


def strip_text_columns(df, text_columns):
    # 6. Strip leading/trailing whitespace to avoid silent mismatches
    print(LINE)
    stripped = []
    missing_cols = []
    for col in text_columns:
        if col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].str.strip()
            stripped.append(col)
        else:
            missing_cols.append(col)

    if stripped:
        print(f"6. Whitespace stripped in: {stripped}")
    if missing_cols:
        print(f"6. Columns not found for whitespace stripping: {missing_cols}")
    print(LINE + "\n")


def categorical_conversion(df, categorical_columns):
    # 7. Convert categorical columns to 'category' dtype
    print(LINE)
    converted = []
    missing_cols = []
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            converted.append(col)
        else:
            missing_cols.append(col)

    if converted:
        print(f"7. Converted to category: {converted}")
    if missing_cols:
        print(f"7. Columns not found for categorical conversion: {missing_cols}")
    print(LINE + "\n")


# (new column, columns it needs, formula, meaning)
DERIVED_FEATURES = [
    ("net_units_sold", ['units_sold', 'returns'],
     lambda d: d['units_sold'] - d['returns'],
     "units_sold - returns"),
    ("price_gap", ['price', 'competitor_pricing'],
     lambda d: d['price'] - d['competitor_pricing'],
     "price - competitor_pricing"),
    ("price_ratio", ['price', 'competitor_pricing'],
     lambda d: d['price'] / d['competitor_pricing'].replace(0, np.nan),
     "price / competitor_pricing"),
    ("inventory_per_visitor", ['inventory_level', 'customer_footfall'],
     lambda d: (d['inventory_level'] / d['customer_footfall'].clip(lower=1)).round(4),
     "inventory_level / customer_footfall (stock available per store visitor)"),
]


def feature_creation(df):
    # 8. Derived columns. net_units_sold is a result of the sale (reporting only, not a model feature);
    #    the others use only what is known before the sale.
    print(LINE)
    created = []
    for name, needed, formula, meaning in DERIVED_FEATURES:
        if all(col in df.columns for col in needed):
            df[name] = formula(df)
            created.append(f"{name} = {meaning}")
        else:
            print(f"8. Missing columns for '{name}': {[c for c in needed if c not in df.columns]}")
    if created:
        print("8. Features created:")
        for line in created:
            print(f"   {line}")
    print(LINE + "\n")


def enforce_dtypes(df, dtype_map):
    # 9. Enforce correct data types on numeric columns
    print(LINE)
    print("9. Enforcing data types:")
    for col, target_dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(target_dtype)
            except ValueError as e:
                print(f"Could not convert '{col}' to {target_dtype}: {e}")
        else:
            print(f"Could not find column '{col}' for dtype conversion.")
    print("Column dtypes after transformation:")
    for col in df.columns:
        print(f"{col:<22} {str(df[col].dtype)}")
    print(LINE + "\n")


def sort_data(df, date_column):
    # 10. Sort rows by date (the model scripts split train / test by date), then by transaction
    print(LINE)
    sort_cols = [c for c in [date_column, 'transaction_id'] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True, kind='stable')
        df.reset_index(drop=True, inplace=True)
        print(f"10. Data sorted by: {sort_cols}")
    else:
        print("10. No columns found for sorting.")
    print(LINE + "\n")


def stock_status_creation(df, under_days=UNDERSTOCK_COVER_DAYS, over_days=OVERSTOCK_COVER_DAYS):
    # 11. Stock status (the label the models predict):
    #       stock_cover = inventory_level / units_sold   (units_sold floored at 1)
    #       Understock if < under_days, Overstock if > over_days, otherwise Healthy.
    #     Both columns are built from the sale itself, so they are results and must never be model features:
    #     the models see the stock and the demand drivers and have to say how the stock compares with demand.
    print(LINE)
    needed = ['inventory_level', 'units_sold']
    if not all(col in df.columns for col in needed):
        print(f"11. Missing columns for stock status: {[c for c in needed if c not in df.columns]}")
        print(LINE + "\n")
        return

    df['stock_cover'] = (df['inventory_level'] / df['units_sold'].clip(lower=1)).round(4)
    df['stock_status'] = np.select(
        [df['stock_cover'] < under_days, df['stock_cover'] > over_days],
        [STOCK_STATUS[0], STOCK_STATUS[2]], default=STOCK_STATUS[1],
    )
    print(f"11. Stock status created: stock_cover, stock_status "
          f"(<{under_days:g} days Understock, >{over_days:g} days Overstock).")
    print(df['stock_status'].value_counts().reindex(STOCK_STATUS).to_string())
    print(LINE + "\n")


def reorder_columns(df):
    # 12. Put columns in a tidy, logical order (any extra columns go at the end)
    print(LINE)
    desired_order = [
        # Time
        'date', 'year', 'quarter', 'month', 'month_name', 'week', 'day', 'weekday', 'day_of_week', 'is_weekend',
        # Product
        'transaction_id', 'product_id', 'category', 'product_subcategory', 'brand',
        # Store
        'store_id', 'store_type', 'region', 'city',
        # Inventory & demand
        'inventory_level', 'units_sold', 'returns', 'net_units_sold', 'stock_cover', 'demand_level',
        # Pricing
        'price', 'discount', 'competitor_pricing', 'price_gap', 'price_ratio',
        # Financials
        'revenue',
        # Context
        'promotion', 'holiday', 'seasonality', 'weather_condition', 'customer_footfall', 'marketing_spend',
        'online_orders', 'economic_index', 'inventory_per_visitor',
        # Supply & service
        'shipping_time', 'supplier_rating', 'customer_satisfaction',
        # Label
        'stock_status',
    ]
    final_cols = [c for c in desired_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in final_cols]
    df_ordered = df[final_cols + extra_cols]
    print(f"12. Columns reordered ({len(final_cols)} placed, {len(extra_cols)} extra kept at the end).")
    print(LINE + "\n")
    return df_ordered


def transform_summary(df):
    # 13. Summary report
    print(LINE)
    print("13. Transformation summary")
    print(f"Rows                    : {len(df):,}")
    print(f"Columns                 : {len(df.columns)}")
    if 'product_id' in df.columns:
        print(f"Unique products     : {df['product_id'].nunique()}")
    if 'store_id' in df.columns:
        print(f"Unique stores       : {df['store_id'].nunique()}")
    if 'region' in df.columns:
        print(f"Unique regions      : {df['region'].nunique()}")
    if 'date' in df.columns:
        print(f"Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
    if 'revenue' in df.columns:
        print(f"Total revenue       : {df['revenue'].sum():,.2f}")

    if all(c in df.columns for c in ['category', 'revenue', 'units_sold']):
        cat_summary = (
            df.groupby('category', observed=True)
              .agg(total_revenue=('revenue', 'sum'), total_units=('units_sold', 'sum'))
              .sort_values('total_revenue', ascending=False)
        )
        print("\nRevenue by category:")
        print(f"{'Category':<18} {'Revenue':>16} {'Units sold':>12}")
        print(f"{'-' * 18} {'-' * 16} {'-' * 12}")
        for cat, row in cat_summary.iterrows():
            print(f"{str(cat):<18} {row['total_revenue']:>16,.2f} {int(row['total_units']):>12,}")
    print(LINE + "\n")


def data_transform(df_clean):
    df_transform = df_clean.copy()

    standardize_columns(df_transform)
    date_conversion(df_transform, 'date')
    numeric_conversion(df_transform, [
        'units_sold', 'price', 'discount', 'inventory_level', 'competitor_pricing', 'customer_footfall',
        'marketing_spend', 'economic_index', 'online_orders', 'returns', 'customer_satisfaction',
        'shipping_time', 'supplier_rating', 'revenue', 'promotion', 'holiday',
    ])
    strip_text_columns(df_transform, [
        'store_id', 'product_id', 'store_type', 'region', 'city', 'category',
        'product_subcategory', 'brand', 'weather_condition', 'seasonality', 'demand_level',
    ])
    categorical_conversion(df_transform, [
        'store_id', 'product_id', 'store_type', 'region', 'city', 'category', 'product_subcategory',
        'brand', 'weather_condition', 'seasonality', 'demand_level',
    ])
    feature_creation(df_transform)
    enforce_dtypes(df_transform, {
        'inventory_level': 'int64', 'units_sold': 'int64', 'returns': 'int64', 'customer_footfall': 'int64',
        'online_orders': 'int64', 'shipping_time': 'int64', 'promotion': 'int64', 'holiday': 'int64',
        'year': 'int64', 'quarter': 'int64', 'month': 'int64', 'day': 'int64', 'week': 'int64',
        'price': 'float64', 'discount': 'float64', 'competitor_pricing': 'float64',
        'revenue': 'float64', 'marketing_spend': 'float64',
    })
    sort_data(df_transform, 'date')
    stock_status_creation(df_transform)
    df_transform = reorder_columns(df_transform)
    transform_summary(df_transform)

    return df_transform
