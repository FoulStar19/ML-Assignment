"""Read the XGBoost results from model/xgboost/outputs/ (for the dashboard / report). Nothing is computed here.

    from model.xgboost.xgboost_results import (load_metrics, load_predictions, load_feature_importance,
                                               load_segment_errors, load_model, predict_units)

    m = load_metrics()                      # dict, see metrics.json
    m["tuned"]["test"]["rmse"]              # also "mae", "mse", "r2", "wape", "bias"
    load_predictions()                      # DataFrame: Date, ID/stock columns (Transaction_ID, Store_ID, Product_ID,
                                            #            Inventory_Level), Units_Sold, pred_tuned, error ...
    predict_units(df)                       # forecast for new rows (see below)

Charts are ready-made PNG files in model/xgboost/outputs/ (list_plots()).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# The one path defined here: where the shared data-preparation code lives, so it can be imported.
BACKEND_DIR = HERE.parents[1] / "src" / "backend"          # model/xgboost/xgboost_results.py -> <project>/src/backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from data_modeling import FEATURES, PREPROCESSOR_FILE, ROOT_DIR, model_output_dir   # noqa: E402
from data_transforming import add_price_features                                    # noqa: E402
if ROOT_DIR not in sys.path:                                                        # so `model.evaluation` can be imported
    sys.path.insert(0, ROOT_DIR)
from model.evaluation import clipped_predict                                        # noqa: E402

OUT_DIR = Path(model_output_dir(HERE.name))                # <project>/model/xgboost/outputs, same folder the training writes to


def _need(name):
    path = OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python model/xgboost/xgboost_model.py")
    return path


def load_metrics() -> dict:
    return json.loads(_need("metrics.json").read_text())


def load_predictions() -> pd.DataFrame:
    """Test-period predictions of the default (pred_default) and tuned (pred_tuned) model next to the actual Units_Sold."""
    return pd.read_csv(_need("predictions.csv"), parse_dates=["Date"])


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(_need("feature_importance.csv"), index_col="feature")


def load_segment_errors() -> pd.DataFrame:
    return pd.read_csv(_need("segment_errors.csv"))


def load_worst_predictions() -> pd.DataFrame:
    return pd.read_csv(_need("worst_predictions.csv"))


def list_plots() -> dict:
    """{plot name: path} for every PNG the training run drew."""
    return {p.stem: p for p in sorted(OUT_DIR.glob("*.png"))}


def load_model():
    return joblib.load(_need("model.joblib"))


def predict_units(df: pd.DataFrame) -> np.ndarray:
    """Forecast Units_Sold for new rows.

    `df` needs the model's input columns (see FEATURES in data_modeling.py): the columns of
    data/processed/retail_store_inventory_processed.csv. If the derived price features (Log_Effective_Price,
    Price_Gap) are not in `df`, they are built from Unit_Price, Discount_Percent and Competitor_Price, with the same
    code the pipeline uses. The pipeline's saved preprocessor does the scaling and one-hot encoding, so the new rows
    are treated exactly like the training rows.
    """
    if any(c not in df.columns for c in FEATURES):
        df = add_price_features(df)
    X = joblib.load(PREPROCESSOR_FILE).transform(df[FEATURES])
    return clipped_predict(load_model(), X)