import argparse
import os

from data_loading import data_load
from data_cleaning import data_clean
from data_transforming import data_transform
from data_modeling import data_model

# ─────────────────────────────────────────────
# 0. FILE PATHS
# ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_FILE = os.path.join(ROOT_DIR, "data", "raw", "retail_store_inventory.csv")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "retail_store_inventory_processed.csv")


def run_pipeline(train_models=False, n_iter=20, cv_splits=3):
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
    # 5. MODEL TRAINING, TUNING AND EVALUATION
    # ─────────────────────────────────────────────
    if train_models:
        data_model(PROCESSED_DATA_FILE, n_iter=n_iter, cv_splits=cv_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load -> clean -> transform -> train the four models")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="random-search candidates per model (default 20)")
    parser.add_argument("--cv-splits", type=int, default=3,
                        help="time-series CV folds used for tuning (default 3)")
    parser.add_argument("--skip-modeling", action="store_true",
                        help="only build the processed CSV, do not train the models")
    args = parser.parse_args()
    run_pipeline(train_models=not args.skip_modeling, n_iter=args.n_iter, cv_splits=args.cv_splits)