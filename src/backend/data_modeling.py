import importlib.util
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

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

# The four model scripts (each one has a run(dataset, n_iter, cv_splits) function)
MODEL_ROOT = Path(__file__).resolve().parents[2] / "model"
MODEL_SCRIPTS = [          # (label, script path)
    ("Logistic Regression", MODEL_ROOT / "logistic_regression" / "logistic_regression_model.py"),
    ("Random Forest",       MODEL_ROOT / "random_forest" / "random_forest_model.py"),
    ("XGBoost",             MODEL_ROOT / "xgboost" / "xgboost_model.py"),
    ("MLP",                 MODEL_ROOT / "mlp" / "mlp_model.py"),
]


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


# ─────────────────────────────────────────────
# 3. TRAIN + EVALUATE ALL MODELS (called by pipeline.py)
# ─────────────────────────────────────────────
def _load_model_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_comparison(results):
    """Print one table with the tuned test-set results of every model that finished."""
    print(LINE)
    print("Model comparison (tuned models, test set)")
    header = (f"{'Model':<32} {'Accuracy':>8} {'Macro-F1':>8} {'Under rec':>9} "
              f"{'Over rec':>8} {'MCC':>7} {'Seconds':>8}")
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        t = res["tuned"]["test"]
        print(f"{name:<32} {t['accuracy']:>8.4f} {t['f1_macro']:>8.4f} "
              f"{t['per_class']['Understock']['recall']:>9.4f} "
              f"{t['per_class']['Overstock']['recall']:>8.4f} {t['mcc']:>7.4f} "
              f"{res['tuned']['seconds']:>8.1f}")
    best = max(results, key=lambda n: results[n]["tuned"]["test"]["f1_macro"])
    print(f"\nBest by test macro-F1 : {best}")
    print(LINE + "\n")


def data_model(processed_data_file=None, n_iter=20, cv_splits=3):
    """Processed CSV -> train, tune and evaluate all four models.

    Every model writes its metrics, plots and reports to model/<name>/outputs/.
    A model that crashes is reported and skipped so the others still run.
    Returns {model name: results dict} for the models that finished.
    """
    print(LINE)
    print("MODEL TRAINING")
    print(f"Random search candidates : {n_iter}   |   time-series CV folds : {cv_splits}")
    print(LINE + "\n")

    dataset = prepare_data(processed_data_file)

    results, failed = {}, []
    for label, script in MODEL_SCRIPTS:
        t0 = time.time()
        try:
            module = _load_model_script(script)
            results[module.MODEL_NAME] = module.run(dataset, n_iter=n_iter, cv_splits=cv_splits)
        except Exception:
            failed.append(label)
            print(f"\n{label} FAILED after {time.time() - t0:.0f}s - continuing with the next model.")
            traceback.print_exc()

    if results:
        model_comparison(results)
    if failed:
        print(f"Models that failed: {failed}\n")
    return results