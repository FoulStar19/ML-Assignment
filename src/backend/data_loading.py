import os
import sys
import numpy as np
import pandas as pd
import plotly as plt

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