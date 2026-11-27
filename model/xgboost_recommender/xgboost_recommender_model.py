import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(HERE))                  # model/<name>/ -> repo root
TRAIN_FILE = os.path.join(ROOT_DIR, "data", "processed", "train.csv")
TEST_FILE = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
MODEL_FILE = os.path.join(HERE, "xgboost_recommender.joblib")

TARGET = "Units_Sold"
NON_FEATURES = [TARGET, "Date", "Transaction_ID", "Product_ID", "Inventory_Level"]   # not model inputs
RANDOM_STATE = 42


def train_xgboost_recommender():
    """Predicts Units_Sold; the recommender compares it with Inventory_Level to flag under/overstock."""
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    X_train = train.drop(columns=NON_FEATURES)
    X_test = test.drop(columns=NON_FEATURES)

    y_train, y_test = train[TARGET], test[TARGET]

    model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=3),
                          scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Best parameters : {search.best_params_}")
    print(f"Best CV MAE     : {-search.best_score_:.4f}")

    # Regression metrics (evaluation.py only covers classifiers)
    y_pred = best.predict(X_test)
    print(f"Test MAE        : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Test RMSE       : {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
    print(f"Test R2         : {r2_score(y_test, y_pred):.4f}")

    joblib.dump(best, MODEL_FILE)
    print(f"Saved model     : {MODEL_FILE}")
    return best
