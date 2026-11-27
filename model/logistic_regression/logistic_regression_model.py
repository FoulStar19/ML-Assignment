import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, auc, roc_auc_score

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[4] # src/backend/model/logistic_regression/<this file>
TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_FILE = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_FILE = ROOT_DIR / "models" / "logistic_regression_model.joblib"

TARGET = "Units_Sold"
NON_FEATURES = [TARGET, "Date", "Transaction_ID", "Product_ID", "Inventory_Level"]   # labels / recommender columns
CLASSES = ["Low", "Medium", "High"] # demand level: bottom / middle / top third of Units_Sold


# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────
def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run pipeline.py first")
    df = pd.read_csv(path)
    return df.drop(columns=NON_FEATURES), df[TARGET]


def demand_cutoffs(units_train):
    # Cut points come from the training data only, so the test set does not influence the labels
    return units_train.quantile([1 / 3, 2 / 3]).tolist()


def to_demand_level(units, cutoffs):
    return pd.cut(units, bins=[-float("inf"), cutoffs[0], cutoffs[1], float("inf")], labels=CLASSES).astype(str)


# ─────────────────────────────────────────────
# 2. LOGISTIC REGRESSION BASELINE
# ─────────────────────────────────────────────
def train_logistic_regression():

    # 1. Load the prepared data (already encoded and scaled by data_modeling)
    print(LINE)
    print("1. Data")
    X_train, units_train = load_split(TRAIN_FILE)
    X_test, units_test = load_split(TEST_FILE)
    cutoffs = demand_cutoffs(units_train)
    y_train = to_demand_level(units_train, cutoffs)
    y_test = to_demand_level(units_test, cutoffs)
    print(f"Train : {len(X_train):,} rows | Test : {len(X_test):,} rows | Inputs : {X_train.shape[1]}")
    print(f"Demand level from {TARGET}: Low <= {cutoffs[0]:.0f}, Medium <= {cutoffs[1]:.0f}, High above")
    print("Train classes:", y_train.value_counts(normalize=True).round(3).reindex(CLASSES).to_dict())
    print(LINE + "\n")

    # 2. Train (plain defaults - this is the baseline the other models have to beat)
    print(LINE)
    print("2. Training")
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    print("LogisticRegression(max_iter=1000) fitted.")
    print(LINE + "\n")

    # 3. Evaluate on the held-out (latest) dates
    print(LINE)
    print("3. Evaluation (test set)")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    majority = y_test.value_counts(normalize=True).max()
    print(f"Accuracy : {acc:.3f}  (always guessing the biggest class: {majority:.3f})")
    print(f"Macro F1 : {f1:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=CLASSES),
                       index=CLASSES, columns=CLASSES).to_string())
    print("\n" + classification_report(y_test, y_pred, labels=CLASSES, digits=3))
    print(LINE + "\n")

    # 4. Save
    print(LINE)
    print("4. Saving")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "cutoffs": cutoffs, "classes": CLASSES}, MODEL_FILE)
    print(f"Model saved : {MODEL_FILE}")
    print(LINE + "\n")

    return model


if __name__ == "__main__":
    train_logistic_regression()