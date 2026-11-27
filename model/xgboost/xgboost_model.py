import os

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model.evaluation import evaluate_model

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(HERE))                  # model/<name>/ -> repo root
TRAIN_FILE = os.path.join(ROOT_DIR, "data", "processed", "train.csv")
TEST_FILE = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
MODEL_FILE = os.path.join(HERE, "xgboost.joblib")

TARGET = "Units_Sold"
NON_FEATURES = [TARGET, "Date", "Transaction_ID", "Product_ID", "Inventory_Level"]   # not model inputs
RANDOM_STATE = 42


def train_xgboost():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    X_train = train.drop(columns=NON_FEATURES)
    X_test = test.drop(columns=NON_FEATURES)

    # High demand = Units_Sold above the TRAIN median (threshold taken from train only)
    threshold = train[TARGET].median()
    y_train = (train[TARGET] > threshold).astype(int)
    y_test = (test[TARGET] > threshold).astype(int)

    # Tune with time-ordered folds (train.csv is sorted by date, so no look-ahead)
    model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=3), scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Best parameters : {search.best_params_}")
    print(f"Best CV F1      : {search.best_score_:.4f}")

    # Evaluate on the held-out (most recent) dates
    y_pred = best.predict(X_test)
    evaluate_model(y_test, y_pred, "XGBoost")
    print(f"Test F1         : {f1_score(y_test, y_pred):.4f}")

    joblib.dump(best, MODEL_FILE)
    print(f"Saved model     : {MODEL_FILE}")
    return best
