from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loading import load_processed
from data_transforming import HORIZON_DAYS, HISTORY_FEATURES, LINE_KEYS, STOCK_STATUS

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_FRACTION = 0.2               # most recent share of the dates is held out as the test set
CLASS_NAMES = list(STOCK_STATUS)  # label 0, 1, 2 = Understock, Healthy, Overstock

# Columns taken straight from the processed CSV. revenue, revenue_per_unit,
# stock_cover and forecast_error are left out on purpose: they are just
# re-arrangements of columns that are already included.
BASE_NUMERIC = [
    "inventory_level", "units_sold", "units_ordered", "demand_forecast",
    "price", "discount", "competitor_pricing", "price_gap",
    "month", "day", "week", "weekday",
]
NUMERIC_FEATURES = BASE_NUMERIC + HISTORY_FEATURES      # history features come from the pipeline
CATEGORICAL_FEATURES = [
    "store_id", "product_id", "category", "region",
    "weather_condition", "seasonality", "holiday_promotion",
]

LINE = "─" * 45


# ─────────────────────────────────────────────
# 1. TRAIN / TEST SETS
# ─────────────────────────────────────────────
@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    numeric_features: list
    categorical_features: list
    class_names: list
    cv_gap: int          # rows to skip between CV train and validation folds (see TimeSeriesSplit)
    split_info: str

    def describe(self):
        def share(y):
            return "  ".join(f"{n} {np.mean(y == i):.1%}" for i, n in enumerate(self.class_names))
        return (
            f"{LINE}\nModel data\n"
            f"Train rows : {len(self.X_train):,}   ({share(self.y_train)})\n"
            f"Test rows  : {len(self.X_test):,}   ({share(self.y_test)})\n"
            f"Features   : {len(self.numeric_features)} numeric + {len(self.categorical_features)} categorical\n"
            f"{self.split_info}\n{LINE}\n"
        )


def prepare_data(data_path=None, test_fraction=TEST_FRACTION, horizon_days=HORIZON_DAYS):
    """Processed CSV -> chronological train / test split (features and label come from the pipeline).

    data_path=None uses data/processed/retail_store_inventory_processed.csv and
    runs the data pipeline first if that file is missing.
    """
    needed = (["date", "stock_status"] + LINE_KEYS + NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    df = load_processed(data_path, required_columns=list(dict.fromkeys(needed)))

    df = df.dropna(subset=["stock_status"]).copy()     # last horizon_days of each line have no label yet
    df["target"] = df["stock_status"].map({n: i for i, n in enumerate(CLASS_NAMES)}).astype(int)
    df["holiday_promotion"] = df["holiday_promotion"].astype(str)   # a 0/1 flag, handled as a category
    df = df.sort_values(["date"] + LINE_KEYS, kind="stable").reset_index(drop=True)   # chronological

    # Chronological split with a purge gap: a training row's label looks `horizon_days`
    # ahead, so training stops that many days before the test period starts.
    dates = pd.Index(df["date"].unique()).sort_values()
    test_start = dates[int(len(dates) * (1 - test_fraction))]
    train_end = test_start - pd.Timedelta(days=horizon_days)
    train = df[df["date"] < train_end]
    test = df[df["date"] >= test_start]

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    rows_per_day = int(train.groupby("date").size().max())
    split_info = (
        f"Train : {train['date'].min().date()} → {(train_end - pd.Timedelta(days=1)).date()}\n"
        f"Test  : {test_start.date()} → {test['date'].max().date()}   "
        f"({horizon_days}-day purge gap, target = stock status over the next {horizon_days} days)"
    )
    return Dataset(
        X_train=train[features].reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_train=train["target"].to_numpy(),
        y_test=test["target"].to_numpy(),
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        class_names=list(CLASS_NAMES),
        cv_gap=rows_per_day * horizon_days,
        split_info=split_info,
    )


# ─────────────────────────────────────────────
# 2. PREPROCESSOR (fitted inside each model's sklearn Pipeline, so CV never sees the validation fold)
# ─────────────────────────────────────────────
def build_preprocessor(numeric_features, categorical_features, scale):
    """Median-impute numbers (+ standardise if scale=True); one-hot encode categories.
    scale=True for Logistic Regression and the MLP; the tree models don't need it."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), list(numeric_features)),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), list(categorical_features)),
        ],
        verbose_feature_names_out=False,
    )
