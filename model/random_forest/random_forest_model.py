import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

# make data_modeling importable when this file is run on its own
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_modeling import load_model_data, model_output_dir, SEED

MODEL_NAME = "xgboost"

# ─────────────────────────────────────────────
# TUNING SETTINGS (edit these to change the tuning)
# ─────────────────────────────────────────────
# Parameters NOT being tuned (fixed for every trial)
def BASE_MODEL():
    return XGBRegressor(random_state=SEED, n_jobs=-1)

# Values to try for each hyperparameter: add / remove values or whole parameters
PARAM_GRID = {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}

SEARCH_METHOD = "grid"                    # "grid" = try every combination, "random" = try N_ITER random ones
N_ITER = 10                               # only used when SEARCH_METHOD = "random"
CV_SPLITS = 3                             # number of time-ordered CV folds (TimeSeriesSplit)
SCORING = "neg_root_mean_squared_error"   # e.g. "neg_mean_absolute_error", "r2"


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def train_xgboost(param_grid=None):
    param_grid = param_grid or PARAM_GRID   # pass a dict here to override PARAM_GRID for one run
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    model = BASE_MODEL()

    # Tune with time-ordered CV on the training rows only (the test set is never used for tuning)
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    if SEARCH_METHOD == "random":
        search = RandomizedSearchCV(model, param_grid, n_iter=N_ITER, cv=cv, scoring=SCORING,
                                    random_state=SEED, n_jobs=-1)
    else:
        search = GridSearchCV(model, param_grid, cv=cv, scoring=SCORING, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")

    # Evaluate on the held-out (most recent) test dates
    train_pred = best.predict(X_train)
    test_pred = best.predict(X_test)
    results = {"model": "XGBoost", "best_params": search.best_params_,
               "cv_score": search.best_score_, "scoring": SCORING,
               "train": evaluate(y_train, train_pred), "test": evaluate(y_test, test_pred)}
    print(f"Train: " + ", ".join(f"{k}={v:.3f}" for k, v in results["train"].items()))
    print(f"Test : " + ", ".join(f"{k}={v:.3f}" for k, v in results["test"].items()))

    # Save model, metrics and test predictions
    out_dir = model_output_dir(MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best, os.path.join(out_dir, f"{MODEL_NAME}.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    preds = meta_test.copy()
    preds["Actual_Units_Sold"] = y_test.values
    preds["Predicted_Units_Sold"] = test_pred
    preds["Error"] = preds["Predicted_Units_Sold"] - preds["Actual_Units_Sold"]
    preds.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    # Feature importance / coefficients for the error and feature analysis
    imp = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])
    print(f"Results saved to {out_dir}")
    return best, results


if __name__ == "__main__":
    train_xgboost()