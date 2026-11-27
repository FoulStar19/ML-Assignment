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
FEATURES = FLAG_COLS + NUM_COLS + CAT_COLS  # the 13 model features (raw columns, before encoding)
NON_FEATURE_COLS = [TARGET, DATE_COL] + STOCK_COLS   # in train.csv / test.csv but never given to a model
TEST_FRACTION = 0.30                        # most recent 30% of dates
SEED = 42                                   # random seed used by every model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
TRAIN_FILE = os.path.join(MODEL_DATA_DIR, "train.csv")
TEST_FILE = os.path.join(MODEL_DATA_DIR, "test.csv")
PREPROCESSOR_FILE = os.path.join(MODEL_DATA_DIR, "preprocessor.joblib")


def model_output_dir(model_name):
    """Where a model writes its results: <project>/model/<model_name>/outputs (e.g. model_output_dir("xgboost"))."""
    return os.path.join(ROOT_DIR, "model", model_name, "outputs")


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


# ─────────────────────────────────────────────
# 5. LOAD THE PREPARED DATA (used by every model script)
# ─────────────────────────────────────────────
def load_train_test():
    """Read train.csv / test.csv (Date parsed) and check the split is a clean, ordered time split.

    Returns the two full DataFrames: model inputs + Units_Sold + Date + ID/stock columns.
    """
    for f in (TRAIN_FILE, TEST_FILE):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found; run `python pipeline.py --skip-training` first.")
    train = pd.read_csv(TRAIN_FILE, parse_dates=[DATE_COL])
    test = pd.read_csv(TEST_FILE, parse_dates=[DATE_COL])
    if get_feature_columns(train) != get_feature_columns(test):
        raise ValueError("train.csv and test.csv have different feature columns; re-run the pipeline.")
    # The split must be by time with no overlap, otherwise the test score would be too optimistic.
    if not train[DATE_COL].is_monotonic_increasing:
        raise ValueError("train.csv is not sorted by date; time-series CV folds would not respect time order.")
    if train[DATE_COL].max() >= test[DATE_COL].min():
        raise ValueError("train and test dates overlap; the split is not a clean time-based split.")
    return train, test


def get_feature_columns(df):
    """The model inputs: every column that is not the target, the date or an ID/stock column."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def load_model_data():
    """train.csv / test.csv split into what a model may see and what it may not.

    Returns X_train, y_train, X_test, y_test (model inputs and Units_Sold) and meta_test
    (Date, Transaction_ID, Store_ID, Product_ID, Inventory_Level of the test rows, for error
    analysis and the recommender). ID, date and stock columns never reach the models.
    """
    train, test = load_train_test()
    features = get_feature_columns(train)
    return (train[features], train[TARGET],
            test[features], test[TARGET],
            test[[DATE_COL] + STOCK_COLS])