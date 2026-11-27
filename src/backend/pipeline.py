import argparse
import os
import sys

from data_loading import data_load
from data_cleaning import data_clean
from data_transforming import data_transform
from data_modeling import data_model

# ─────────────────────────────────────────────
# 0. FILE PATHS
# ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
RAW_DATA_FILE = os.path.join(ROOT_DIR, "data", "raw", "retail_sales_forecast_dataset.csv")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "retail_store_inventory_processed.csv")


def run_pipeline(run_training=True, outlier_action="report"):
    # Create the output folder if it doesn't exist yet - just in case
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # ─────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────
    df_loaded = data_load(RAW_DATA_FILE)

    # ─────────────────────────────────────────────
    # 2. DATA CLEANING
    # ─────────────────────────────────────────────
    df_cleaned = data_clean(df_loaded, outlier_action=outlier_action)

    # ─────────────────────────────────────────────
    # 3. DATA TRANSFORMATION
    # ─────────────────────────────────────────────
    df_transformed = data_transform(df_cleaned)

    # ─────────────────────────────────────────────
    # 4. SAVE PROCESSED DATA
    # ─────────────────────────────────────────────
    df_transformed.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_FILE}\n")

    # ─────────────────────────────────────────────
    # 5. PREPARE FOR MODELING (train/test split, encoding, scaling)
    # ─────────────────────────────────────────────
    data_model(PROCESSED_DATA_FILE)

    # ─────────────────────────────────────────────
    # 6. TRAIN MODELS
    # ─────────────────────────────────────────────
    if run_training:
        from data_training import train_models
        train_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load -> clean -> transform -> split/encode -> train the models")
    parser.add_argument("--skip-training", "--skip-modeling", dest="skip_training", action="store_true",
                        help="build the processed CSV and the train/test files, but do not train the models")
    parser.add_argument("--outlier-action", choices=["report", "cap"], default="report",
                        help="report outliers only (default) or cap them in the input columns")
    args = parser.parse_args()
    run_pipeline(run_training=not args.skip_training, outlier_action=args.outlier_action)