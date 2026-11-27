import os

import joblib
from sklearn.linear_model import LinearRegression

from data_modeling import load_model_data
from model.evaluation import evaluate_model

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(MODEL_DIR, "linear_regression.joblib")


def train_linear_regression(show_plots=False):
    """Baseline for the other models: ordinary least squares on the prepared train/test files.

    Plain linear regression has no hyperparameters to tune; the other three models are tuned
    individually and compared against this one.
    """
    X_train, y_train, X_test, y_test, _ = load_model_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Linear Regression (baseline)", plots=show_plots)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return model, metrics