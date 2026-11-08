import os

from data_loading import data_load
from data_cleaning import data_clean

# ─────────────────────────────────────────────
# 0. FILE PATHS
# ─────────────────────────────────────────────
RAW_DATA_FILE = os.path.join("data", "raw", "retail_store_inventory.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "retail_store_inventory_processed.csv")

# Create the output folder if it doesn't exist yet - just in case
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df_loaded = data_load(RAW_DATA_FILE)

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
df_cleaned = data_clean(df_loaded)
