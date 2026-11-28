import os
import sys
import json
from pathlib import Path

import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")                      # save PNGs only; no pop-up windows blocking the pipeline
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance

# Project paths
HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent.parent
BACKEND_DIR = ROOT_DIR / "src" / "backend"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from data_modeling import SEED, TARGET, NON_FEATURE_COLS
from model.evaluation import (
    clipped_predict,
    evaluate_model,
)

MODEL_NAME = "Random Forest"
N_SPLITS = 5                      # same time-ordered CV folds as the other three models
OUT_DIR = HERE / "outputs"
MODEL_FILE = HERE / "random_forest.joblib"


def save_regression_plots(y_true, y_pred, out_dir):
    y_true = pd.Series(y_true).to_numpy()
    y_pred = pd.Series(y_pred).to_numpy()

    residuals = y_pred - y_true

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3)

    top = max(y_true.max(), y_pred.max())
    ax.plot([0, top], [0, top], "k--")

    ax.set(
        title="Actual vs Predicted - Random Forest",
        xlabel="Actual Units_Sold",
        ylabel="Predicted Units_Sold"
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=300)
    plt.close(fig)

    # 2. Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=8, alpha=0.3)
    ax.axhline(0, color="black", linestyle="--")

    ax.set(
        title="Residuals vs Predicted - Random Forest",
        xlabel="Predicted Units_Sold",
        ylabel="Predicted - Actual"
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_vs_predicted.png", dpi=300)
    plt.close(fig)

    # 3. Residual Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40)

    ax.set(
        title="Residual Distribution - Random Forest",
        xlabel="Predicted - Actual",
        ylabel="Rows"
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_distribution.png", dpi=300)
    plt.close(fig)


def save_impurity_importance(model, feature_names, out_dir):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    importance.to_csv(out_dir / "impurity_importance.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(15).sort_values().plot.barh(ax=ax)

    ax.set(
        title="Impurity Feature Importance - Random Forest",
        xlabel="Importance"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "impurity_importance.png", dpi=300)
    plt.close(fig)


def save_permutation_importance(model, X_test, y_test, out_dir):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1
    )

    importance = pd.Series(
        result.importances_mean,
        index=X_test.columns
    ).sort_values(ascending=False)

    importance.to_csv(out_dir / "permutation_importance.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(15).sort_values().plot.barh(ax=ax)

    ax.set(
        title="Permutation Feature Importance - Random Forest",
        xlabel="Increase in RMSE when shuffled"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=300)
    plt.close(fig)



def train_random_forest():

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 50)
    print("RANDOM FOREST REGRESSION")
    print("=" * 50)

    # 1. Load shared training and testing data
    train = pd.read_csv(ROOT_DIR / "data" / "processed" / "train.csv")
    test = pd.read_csv(ROOT_DIR / "data" / "processed" / "test.csv")

    X_train = train.drop(columns=NON_FEATURE_COLS)
    y_train = train[TARGET]

    X_test = test[X_train.columns]
    y_test = test[TARGET]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows : {len(X_test):,}")
    print(f"Features     : {X_train.shape[1]}")

    # 2. Build Random Forest Regressor
    model = RandomForestRegressor(
        random_state=SEED,
        n_jobs=1
    )

    # 3. Hyperparameter tuning
    param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
    }
    
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",   # same tuning metric as the other three models
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting hyperparameter tuning...")

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)

    print(f"Best CV RMSE: {-search.best_score_:.4f}")

    # 4. Predict on test data
    y_pred = clipped_predict(best_model, X_test)

    # 5. Evaluate regression performance
    metrics = evaluate_model(
        y_test,
        y_pred,
        MODEL_NAME,
        plots=False                       # the PNGs are saved to outputs/ below instead
    )

    # 6. Save evaluation metrics
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 7. Save cross-validation results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(
        OUT_DIR / "cv_results.csv",
        index=False
    )

    # Save hyperparameter tuning graph
    top_results = cv_results.nsmallest(10, "rank_test_score").copy()

    top_results["RMSE"] = -top_results["mean_test_score"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        range(len(top_results)),
        top_results["RMSE"]
    )

    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels([
        f"Rank {rank}" for rank in top_results["rank_test_score"]
    ])

    ax.invert_yaxis()
    ax.set(
        title="Top 10 Hyperparameter Combinations",
        xlabel="Cross-validation RMSE",
        ylabel="Parameter Combination"
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "hyperparameter_tuning.png", dpi=300)
    plt.close(fig)

    # 8. Feature importance
    importance = pd.Series(
        best_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    importance.to_csv(
        OUT_DIR / "feature_importance.csv",
        header=["importance"]
    )

    # Save regression evaluation graphs
    save_regression_plots(y_test, y_pred, OUT_DIR)

    # Save impurity feature importance
    save_impurity_importance(
        best_model,
        X_train.columns,
        OUT_DIR
    )

    # Save permutation feature importance
    save_permutation_importance(
        best_model,
        X_test,
        y_test,
        OUT_DIR
    )

    # 9. Save trained model
    joblib.dump(best_model, MODEL_FILE)

    print("\n" + "=" * 50)
    print("RANDOM FOREST TRAINING COMPLETED")
    print("=" * 50)
    print(f"Model saved  : {MODEL_FILE}")
    print(f"Results saved: {OUT_DIR}")

    return best_model


if __name__ == "__main__":
    train_random_forest()