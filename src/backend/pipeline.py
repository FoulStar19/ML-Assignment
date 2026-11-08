import os

from data_loading import data_load
from data_cleaning import data_clean
from data_transforming import data_transform

# ─────────────────────────────────────────────
# 0. FILE PATHS
# ─────────────────────────────────────────────
RAW_DATA_FILE = os.path.join("data", "raw", "retail_store_inventory.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "retail_store_inventory_processed.csv")

# Create the output folder if it doesn't exist yet - just in case
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def run_pipeline():
    # ─────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────
    df_loaded = data_load(RAW_DATA_FILE)

    # ─────────────────────────────────────────────
    # 2. DATA CLEANING
    # ─────────────────────────────────────────────
    df_cleaned = data_clean(df_loaded)

    # ─────────────────────────────────────────────
    # 3. DATA TRANSFORMATION
    # ─────────────────────────────────────────────
    df_transformed = data_transform(df_cleaned)

    # ─────────────────────────────────────────────
    # 4. SAVE PROCESSED DATA
    # ─────────────────────────────────────────────
    df_transformed.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_FILE}")
