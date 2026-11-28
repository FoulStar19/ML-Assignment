"""Paths, model list and default inventory rules for the dashboard. Column names come from the backend code."""
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]          # src/frontend/config.py -> project root
for p in (ROOT_DIR, ROOT_DIR / "src" / "backend"):       # so `data_modeling` and `model.*` can be imported
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data_modeling as _dm  # noqa: E402
from data_modeling import DATE_COL, FEATURES, STOCK_COLS, TARGET, get_feature_columns  # noqa: E402,F401

TRAIN_FILE, TEST_FILE, PREPROCESSOR_FILE = Path(_dm.TRAIN_FILE), Path(_dm.TEST_FILE), Path(_dm.PREPROCESSOR_FILE)

PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "retail_store_inventory_processed.csv"
RAW_FILE = ROOT_DIR / "data" / "raw" / "retail_sales_forecast_dataset.csv"   # used for the data explorer and Shipping_Time
MODEL_DIR = ROOT_DIR / "model"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: Path
    kind: str      # "plain" = a fitted estimator, "linear" / "mlp" = the dict saved by that model script


MODELS = [
    ModelSpec("Linear Regression", MODEL_DIR / "linear_regression" / "linear_regression_model.joblib", "linear"),
    ModelSpec("Random Forest", MODEL_DIR / "random_forest" / "random_forest.joblib", "plain"),
    ModelSpec("XGBoost", MODEL_DIR / "xgboost" / "outputs" / "model.joblib", "plain"),
    ModelSpec("MLP", MODEL_DIR / "mlp" / "mlp_model.joblib", "mlp"),
]

# Default inventory rules (all adjustable in the sidebar)
SERVICE_LEVEL = 95       # % chance of not running out while waiting for a delivery (sets the safety stock)
LEAD_DAYS = 4            # fixed lead time (days) when Shipping_Time is not used; 4 is the median Shipping_Time
REVIEW_DAYS = 7          # extra days of demand the stock must also cover after the lead time
TARGET_DAYS = 21         # replenish up to this many days of demand (plus safety stock)
OVERSTOCK_DAYS = 40      # stock covers more days than this -> overstocking
HIGH_DEMAND_PCT = 90     # "high demand" = predicted units above this percentile of past sales
ABC_CUTS = (0.70, 0.90)  # ABC classes: items making up the first 70% / next 20% / last 10% of revenue

STATUS_COLORS = {"Understocking": "#d62728", "Overstocking": "#ff7f0e", "Sufficient stock": "#2ca02c"}
STATUS_ICONS = {"Understocking": "🔴", "Overstocking": "🟠", "Sufficient stock": "🟢"}