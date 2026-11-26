import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loading import load_processed
from data_transforming import DATE_COL, TARGET, FLAG_COLS, NUM_COLS, CAT_COLS, STOCK_COLS   # column names defined once, in the transform step

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS (column names come from data_transforming)
# ─────────────────────────────────────────────
FEATURES = FLAG_COLS + NUM_COLS + CAT_COLS  # the 11 model features
TEST_FRACTION = 0.30                        # most recent 30% of dates

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
TRAIN_FILE = os.path.join(MODEL_DATA_DIR, "train.csv")
TEST_FILE = os.path.join(MODEL_DATA_DIR, "test.csv")
PREPROCESSOR_FILE = os.path.join(MODEL_DATA_DIR, "preprocessor.joblib")


# ─────────────────────────────────────────────
# 4. DATA MODELING (prepare the data, no training)
# ─────────────────────────────────────────────
def data_model(processed_data_file):

    df = load_processed(processed_data_file)

    # 1. Time-based split 
    print(LINE)
    print("1. Train / test split")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if not df[DATE_COL].is_monotonic_increasing:
        raise ValueError(f"Processed data is not sorted by {DATE_COL}; re-run the pipeline so data_transform sorts it.")
    cutoff = df[DATE_COL].iloc[int(len(df) * (1 - TEST_FRACTION))]
    train = df[df[DATE_COL] < cutoff].reset_index(drop=True)    # whole days only, none split across train/test
    test = df[df[DATE_COL] >= cutoff].reset_index(drop=True)
    print(f"Train : {len(train):,} rows  ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()})")
    print(f"Test  : {len(test):,} rows  ({test[DATE_COL].min().date()} to {test[DATE_COL].max().date()})")
    print("Latest dates are held out; both files stay in date order, as TimeSeriesSplit needs.")
    print(LINE + "\n")

    # 2. Encode and scale (fitted on the training rows only, so nothing leaks from the test set)
    print(LINE)
    print("2. Encoding")
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("flag", "passthrough", FLAG_COLS),
    ], verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")
    X_train = preprocessor.fit_transform(train[FEATURES])
    X_test = preprocessor.transform(test[FEATURES])
    print(f"{len(FEATURES)} features -> {X_train.shape[1]} model inputs "
          f"(numbers scaled, categories one-hot, flags kept as 0/1).")
    print(LINE + "\n")

    # 3. Save
    print(LINE)
    print("3. Saving")
    os.makedirs(MODEL_DATA_DIR, exist_ok=True)
    train_out = pd.concat([X_train, train[[TARGET, DATE_COL] + STOCK_COLS]], axis=1)
    test_out = pd.concat([X_test, test[[TARGET, DATE_COL] + STOCK_COLS]], axis=1)
    train_out.to_csv(TRAIN_FILE, index=False)
    test_out.to_csv(TEST_FILE, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"Train set    : {TRAIN_FILE}")
    print(f"Test set     : {TEST_FILE}")
    print(f"Preprocessor : {PREPROCESSOR_FILE}")
    print(f"Columns      : {X_train.shape[1]} inputs + {TARGET}, {DATE_COL} and {len(STOCK_COLS)} stock columns")
    print(LINE + "\n")

    return train_out, test_out