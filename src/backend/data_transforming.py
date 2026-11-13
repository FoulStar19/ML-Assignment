import os
import numpy as np
import pandas as pd

LINE = "─" * 45

# Settings for the model-ready columns (steps 15-16). The model scripts import these too.
HORIZON_DAYS = 7                 # how far ahead the stock status is judged
UNDERSTOCK_COVER_DAYS = 1.0      # forward cover below this -> Understock (dashboard default lead time)
OVERSTOCK_COVER_DAYS = 3.0       # forward cover above this -> Overstock  (dashboard default overstock days)
STOCK_STATUS = ["Understock", "Healthy", "Overstock"]
LINE_KEYS = ['store_id', 'product_id']   # one "line" = one product in one store
HISTORY_FEATURES = [
    'sales_lag1', 'sales_lag7', 'sales_roll7_mean', 'sales_roll14_mean', 'sales_roll7_std',
    'forecast_roll7_mean', 'forecast_error_roll7_mean', 'inventory_change_1d',
    'cover_days_recent', 'cover_days_forecast',
]

# ─────────────────────────────────────────────
# 3. DATA TRANSFORMATION
# ─────────────────────────────────────────────
def standardize_columns(df):
    # 1. Make column names lowercase snake_case, e.g. 'Store ID' -> 'store_id'
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    print(LINE)
    print("1. Column names standardized to lowercase snake_case.")
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

    # 4. Create date features: year, month, day, weekday, month_name, week, day_of_week
    print(LINE)
    if date_column in df.columns:
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['weekday'] = df[date_column].dt.weekday
        df['month_name'] = df[date_column].dt.month_name()
        df['week'] = df[date_column].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df[date_column].dt.day_name()
        print("4. Date features created: year, month, day, weekday, month_name, week, day_of_week.")
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


def feature_creation(df):
    print(LINE)

    # 8. Revenue = units_sold * price * (1 - discount / 100)
    needed = ['units_sold', 'price', 'discount']
    if all(col in df.columns for col in needed):
        df['revenue'] = (df['units_sold'] * df['price'] * (1 - df['discount'] / 100)).round(2)
        print("8. Feature created: revenue.")
    else:
        print(f"8. Missing columns for 'revenue': {[c for c in needed if c not in df.columns]}")

    # 9. Revenue per unit = revenue / units_sold (0 where nothing was sold)
    needed = ['revenue', 'units_sold']
    if all(col in df.columns for col in needed):
        df['revenue_per_unit'] = np.where(
            df['units_sold'] > 0,
            (df['revenue'] / df['units_sold'].replace(0, np.nan)).round(4),
            0.0,
        )
        print("9. Feature created: revenue_per_unit.")
    else:
        print(f"9. Missing columns for 'revenue_per_unit': {[c for c in needed if c not in df.columns]}")

    # 10. Forecast error = units_sold - demand_forecast
    needed = ['units_sold', 'demand_forecast']
    if all(col in df.columns for col in needed):
        df['forecast_error'] = df['units_sold'] - df['demand_forecast']
        print("10. Feature created: forecast_error.")
    else:
        print(f"10. Missing columns for 'forecast_error': {[c for c in needed if c not in df.columns]}")

    # 11. Price gap vs competitor = price - competitor_pricing
    needed = ['price', 'competitor_pricing']
    if all(col in df.columns for col in needed):
        df['price_gap'] = df['price'] - df['competitor_pricing']
        print("11. Feature created: price_gap.")
    else:
        print(f"11. Missing columns for 'price_gap': {[c for c in needed if c not in df.columns]}")

    # 12. Stock cover = inventory_level / units_sold (avoid divide-by-zero)
    needed = ['inventory_level', 'units_sold']
    if all(col in df.columns for col in needed):
        df['stock_cover'] = df['inventory_level'] / df['units_sold'].replace(0, np.nan)
        print("12. Feature created: stock_cover.")
    else:
        print(f"12. Missing columns for 'stock_cover': {[c for c in needed if c not in df.columns]}")
    
    print(LINE + "\n")


def enforce_dtypes(df, dtype_map):
    # 13. Enforce correct data types on numeric columns
    print(LINE)
    print("13. Enforcing data types:")
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
    # 14. Sort rows by store, product and date
    print(LINE)
    sort_cols = [c for c in ['store_id', 'product_id', date_column] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"14. Data sorted by: {sort_cols}")
    else:
        print("14. No columns found for sorting.")
    print(LINE + "\n")

def history_feature_creation(df):
    # 15. History features per store-product line. Every window ends on the current day, never later.
    #     Rows must already be sorted by store, product, date (step 14).
    print(LINE)
    needed = LINE_KEYS + ['units_sold', 'demand_forecast', 'forecast_error', 'inventory_level']
    if not all(col in df.columns for col in needed):
        print(f"15. Missing columns for history features: {[c for c in needed if c not in df.columns]}")
        print(LINE + "\n")
        return

    g = df.groupby(LINE_KEYS, sort=False, observed=True)
    df['sales_lag1'] = g['units_sold'].shift(1)
    df['sales_lag7'] = g['units_sold'].shift(7)
    df['sales_roll7_mean'] = g['units_sold'].transform(lambda s: s.rolling(7, min_periods=1).mean()).round(4)
    df['sales_roll14_mean'] = g['units_sold'].transform(lambda s: s.rolling(14, min_periods=1).mean()).round(4)
    df['sales_roll7_std'] = g['units_sold'].transform(lambda s: s.rolling(7, min_periods=2).std()).round(4)
    df['forecast_roll7_mean'] = g['demand_forecast'].transform(lambda s: s.rolling(7, min_periods=1).mean()).round(4)
    df['forecast_error_roll7_mean'] = g['forecast_error'].transform(lambda s: s.rolling(7, min_periods=1).mean()).round(4)
    df['inventory_change_1d'] = df['inventory_level'] - g['inventory_level'].shift(1)
    # Days of cover on recent demand - the same idea as the dashboard's days_of_cover
    df['cover_days_recent'] = (df['inventory_level'] / df['sales_roll7_mean'].clip(lower=1)).round(4)
    df['cover_days_forecast'] = (df['inventory_level'] / df['forecast_roll7_mean'].clip(lower=1)).round(4)
    print(f"15. History features created: {HISTORY_FEATURES}")
    print(LINE + "\n")


def stock_status_creation(df, horizon_days=HORIZON_DAYS,
                          under_days=UNDERSTOCK_COVER_DAYS, over_days=OVERSTOCK_COVER_DAYS):
    # 16. Stock status over the NEXT horizon_days days (the label the models predict):
    #       forward_cover_days = inventory_level / average units_sold over days t+1 .. t+horizon
    #       Understock if < under_days, Overstock if > over_days, otherwise Healthy.
    #     The last horizon_days rows of each line have no complete future window -> left empty.
    #     This column looks into the FUTURE, so it must never be used as a model feature.
    print(LINE)
    needed = LINE_KEYS + ['units_sold', 'inventory_level']
    if not all(col in df.columns for col in needed):
        print(f"16. Missing columns for stock status: {[c for c in needed if c not in df.columns]}")
        print(LINE + "\n")
        return

    g = df.groupby(LINE_KEYS, sort=False, observed=True)
    future_demand = g['units_sold'].transform(
        lambda s: s.rolling(horizon_days).mean().shift(-horizon_days)
    )
    df['forward_cover_days'] = (df['inventory_level'] / future_demand.clip(lower=1)).round(4)
    status = np.select(
        [df['forward_cover_days'] < under_days, df['forward_cover_days'] > over_days],
        [STOCK_STATUS[0], STOCK_STATUS[2]], default=STOCK_STATUS[1],
    ).astype(object)
    status[future_demand.isna().to_numpy()] = np.nan
    df['stock_status'] = status
    print(f"16. Stock status created: forward_cover_days, stock_status "
          f"(next {horizon_days} days; <{under_days:g} Understock, >{over_days:g} Overstock).")
    print(df['stock_status'].value_counts(dropna=False).to_string())
    print(LINE + "\n")


def reorder_columns(df):
    # 17. Put columns in a tidy, logical order (any extra columns go at the end)
    print(LINE)
    desired_order = [
        # Time
        'date', 'year', 'month', 'month_name', 'day', 'week', 'weekday', 'day_of_week',
        # Product & location
        'product_id', 'category', 'store_id', 'region',
        # Inventory & demand
        'inventory_level', 'units_sold', 'units_ordered', 'demand_forecast',
        'forecast_error', 'stock_cover',
        # Pricing
        'price', 'discount', 'competitor_pricing', 'price_gap',
        # Financials
        'revenue', 'revenue_per_unit',
        # Context
        'weather_condition', 'holiday_promotion', 'seasonality',
        # Model-ready columns (steps 15-16)
        *HISTORY_FEATURES, 'forward_cover_days', 'stock_status',
    ]
    final_cols = [c for c in desired_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in final_cols]
    df_ordered = df[final_cols + extra_cols]
    print(f"17. Columns reordered ({len(final_cols)} placed, {len(extra_cols)} extra kept at the end).")
    print(LINE + "\n")
    return df_ordered


def transform_summary(df):
    # 18. Summary report
    print(LINE)
    print("18. Transformation summary")
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
        'inventory_level', 'units_sold', 'units_ordered',
        'demand_forecast', 'price', 'discount', 'competitor_pricing',
    ])
    strip_text_columns(df_transform, [
        'store_id', 'product_id', 'category', 'region',
        'weather_condition', 'seasonality',
    ])
    categorical_conversion(df_transform, [
        'store_id', 'product_id', 'category', 'region',
        'weather_condition', 'holiday_promotion', 'seasonality',
    ])
    feature_creation(df_transform)
    enforce_dtypes(df_transform, {
        'inventory_level': 'int64', 'units_sold': 'int64', 'units_ordered': 'int64',
        'year': 'int64', 'month': 'int64', 'day': 'int64', 'week': 'int64',
        'price': 'float64', 'discount': 'float64', 'competitor_pricing': 'float64',
        'demand_forecast': 'float64',
    })
    sort_data(df_transform, 'date')
    history_feature_creation(df_transform)
    stock_status_creation(df_transform)
    df_transform = reorder_columns(df_transform)
    transform_summary(df_transform)

    return df_transform