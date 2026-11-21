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
from data_transforming import STOCK_STATUS

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_FRACTION = 0.3               # most recent share of the dates is held out as the test set
CLASS_NAMES = list(STOCK_STATUS)  # label 0, 1, 2 = Understock, Healthy, Overstock

# Model inputs: what is known (or planned) before the sale - the stock on hand, price and promotion
# settings, store traffic and marketing, supplier details and the calendar.
NUMERIC_FEATURES = [
    "inventory_level",
    "price", "discount", "competitor_pricing", "price_gap", "price_ratio",
    "promotion", "holiday",
    "customer_footfall", "marketing_spend", "online_orders", "economic_index",
    "shipping_time", "supplier_rating", "inventory_per_visitor",
    "month", "week", "day", "weekday", "is_weekend",
]
CATEGORICAL_FEATURES = [
    "store_type", "region", "category", "product_subcategory", "brand",
    "weather_condition", "seasonality",
]
# Left out on purpose:
#   units_sold, returns, net_units_sold, revenue, demand_level, customer_satisfaction - results of the sale
#   stock_cover, stock_status - the label is built from inventory_level and units_sold
#   store_id, product_id - labels, not stable stores / products in this data (one ID is sold under many
#                          categories and store types); adding them made the models worse in testing
#   city - no gain over region        year - the test period is a year the model has never seen

LINE = "─" * 45

# The four model scripts (each one has a run(dataset, n_iter, cv_splits) function)
MODEL_ROOT = Path(__file__).resolve().parents[2] / "model"
MODEL_SCRIPTS = [          # (label, script path)
    ("Logistic Regression", MODEL_ROOT / "logistic_regression" / "logistic_regression_model.py"),
    ("Random Forest",       MODEL_ROOT / "random_forest" / "random_forest_model.py"),
    ("XGBoost",             MODEL_ROOT / "xgboost" / "xgboost_model.py"),
    ("MLP",                 MODEL_ROOT / "mlp" / "mlp_model.py"),
]
# Fifth model: XGBoost regressor that forecasts demand; the dashboard's recommender uses it to size orders
DEMAND_SCRIPT = MODEL_ROOT / "xgboost_demand" / "xgboost_demand_model.py"


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


def _split_by_date(df, class_names, target_text, test_fraction):
    """Chronological split: the most recent `test_fraction` of the dates is the test set.

    The label and the features all come from the same record (nothing looks into the future),
    so no purge gap is needed between the train and test periods (cv_gap = 0).
    """
    df = df.sort_values(["date", "transaction_id"], kind="stable").reset_index(drop=True)
    dates = pd.Index(df["date"].unique()).sort_values()
    test_start = dates[int(len(dates) * (1 - test_fraction))]
    train = df[df["date"] < test_start]
    test = df[df["date"] >= test_start]

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    split_info = (
        f"Train : {train['date'].min().date()} → {train['date'].max().date()}\n"
        f"Test  : {test['date'].min().date()} → {test['date'].max().date()}   "
        f"(most recent {test_fraction:.0%} of the dates, target = {target_text})"
    )
    return Dataset(
        X_train=train[features].reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_train=train["target"].to_numpy(),
        y_test=test["target"].to_numpy(),
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        class_names=class_names,
        cv_gap=0,
        split_info=split_info,
    )


def prepare_data(data_path=None, test_fraction=TEST_FRACTION):
    """Processed CSV -> chronological train / test split (features and label come from the pipeline).

    data_path=None uses data/processed/retail_store_inventory_processed.csv and
    runs the data pipeline first if that file is missing.
    """
    needed = ["date", "transaction_id", "stock_status"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = load_processed(data_path, required_columns=list(dict.fromkeys(needed)))

    df = df.dropna(subset=["stock_status"]).copy()
    df["target"] = df["stock_status"].map({n: i for i, n in enumerate(CLASS_NAMES)}).astype(int)
    return _split_by_date(df, list(CLASS_NAMES), "stock status of the record", test_fraction)


def prepare_demand_data(data_path=None, test_fraction=TEST_FRACTION):
    """Same features and chronological split as prepare_data, but the target is a number:
    the units sold in the record (what the stock-status label compares the stock with)."""
    needed = ["date", "transaction_id", "units_sold"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = load_processed(data_path, required_columns=list(dict.fromkeys(needed)))

    df = df.dropna(subset=["units_sold"]).copy()
    df["target"] = df["units_sold"].astype(float)
    return _split_by_date(df, [], "units sold in the record", test_fraction)


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
    """Processed CSV -> train, tune and evaluate the four classifiers and the XGBoost demand model.

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

    # The demand model is not a classifier, so it is trained on its own dataset and kept out of the comparison table.
    t0 = time.time()
    try:
        _load_model_script(DEMAND_SCRIPT).run(prepare_demand_data(processed_data_file), n_iter=n_iter, cv_splits=cv_splits)
    except Exception:
        failed.append("XGBoost demand")
        print(f"\nXGBoost demand FAILED after {time.time() - t0:.0f}s.")
        traceback.print_exc()

    if failed:
        print(f"Models that failed: {failed}\n")
    return results