import os
import numpy as np
import pandas as pd

LINE = "─" * 45

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def data_clean(df_load):
    df_clean = df_load.copy()
 
    # 1. Check for Missing Values
    print(LINE)
    print("1. Missing values")
    missing = df_clean.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        report = pd.DataFrame({
            "missing": missing,
            "percent": (missing / len(df_clean) * 100).round(2),
        })
        print(report.to_string())
        print(f"\nTotal missing cells : {int(missing.sum()):,}")
        print(f"Rows with any missing : {int(df_clean.isna().any(axis=1).sum()):,}")
    print(LINE + "\n")
 
    # 2. Check for Duplicate Rows
    print(LINE)
    print("2. Duplicate rows")
    n_dup = int(df_clean.duplicated().sum())
    if n_dup == 0:
        print("No duplicate rows found.")
    else:
        print(f"Duplicate rows : {n_dup:,} ({n_dup / len(df_clean) * 100:.2f}%)")
        print("First few duplicates:")
        print(df_clean[df_clean.duplicated(keep="first")].head().to_string())
    print(LINE + "\n")
 
    return df_clean