import os
import pandas as pd

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
# Check if the raw data file exists
def data_load(raw_data_file):
    if not os.path.exists(raw_data_file):
        raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")

    df = pd.read_csv(raw_data_file)

    print("─────────────────────────────────────────────")
    print("Raw data loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(f"Columns : {list(df.columns)}")
    print("─────────────────────────────────────────────\n")

    return df

# ─────────────────────────────────────────────
# 1b. LOAD PROCESSED DATA (used by the model scripts)
# ─────────────────────────────────────────────
def load_processed(processed_data_file=None, required_columns=()):
    from pipeline import PROCESSED_DATA_FILE, run_pipeline

    path = processed_data_file or PROCESSED_DATA_FILE
    use_default = processed_data_file is None

    if not os.path.exists(path):
        if not use_default:
            raise FileNotFoundError(f"Processed data file not found: {path}")
        print("Processed data not found - running the data pipeline first...\n")
        run_pipeline()

    df = pd.read_csv(path, parse_dates=["date"])

    absent = [c for c in required_columns if c not in df.columns]
    if absent and use_default:
        print(f"Processed data is out of date (missing {absent}) - re-running the data pipeline...\n")
        run_pipeline()
        df = pd.read_csv(path, parse_dates=["date"])
        absent = [c for c in required_columns if c not in df.columns]
    if absent:
        raise ValueError(f"Processed data is missing columns {absent}. "
                         "Re-run src/backend/pipeline.py to rebuild it.")
    return df