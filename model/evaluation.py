import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Regression evaluation for Units_Sold, shared by every model. Matplotlib is imported inside the plot
# functions, so importing this module for the metrics alone (e.g. from a dashboard) stays light.
#
# Sign convention everywhere: error = predicted - actual, so a NEGATIVE bias means the model
# under-predicts (stock-out risk).


def clipped_predict(model, X):
    """model.predict, with negative values set to 0 (units sold cannot be negative)."""
    return np.clip(model.predict(X), 0, None)


def regression_metrics(y_true, y_pred):
    """MAE, MSE, RMSE and R² (the four the assignment asks for) + WAPE and bias (useful for stock planning).

    WAPE = total absolute error / total units sold ("we miss about X% of demand").
    bias = average (prediction - actual): negative means the model under-predicts.
    """
    y, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = pred - y
    mse = float(np.mean(err ** 2))
    return {"mae": float(np.mean(np.abs(err))), "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y, pred)), "wape": float(np.abs(err).sum() / y.sum()), "bias": float(err.mean())}


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3)
    top = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, top], [0, top], "k--", linewidth=0.8)
    ax.grid(True)
    ax.set(title=f"Actual vs Predicted for {model_name}", xlabel="Actual Units_Sold", ylabel="Predicted Units_Sold")
    plt.show()


def plot_residuals(y_true, y_pred, model_name):
    import matplotlib.pyplot as plt
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, s=6, alpha=0.3)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set(title=f"Residuals vs Predicted ({model_name})", xlabel="Predicted Units_Sold", ylabel="Predicted - Actual")
    axes[1].hist(residuals, bins=40)
    axes[1].set(title="Residual distribution", xlabel="Predicted - Actual", ylabel="Rows")
    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, feature_names, model_name, top_n=15):
    """Bar chart of the top_n features. For linear models pass absolute coefficients."""
    import matplotlib.pyplot as plt
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    series.iloc[::-1].plot.barh(ax=ax)
    ax.set(title=f"Feature importance for {model_name}", xlabel="Importance")
    plt.tight_layout()
    plt.show()
    return series


def plot_loss_curve(train_losses, val_losses, model_name):
    """For the MLP (loss per epoch / iteration)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(train_losses, label="Train Loss")
    if val_losses is not None:
        ax.plot(val_losses, label="Validation Loss")
    ax.set(title=f"Loss Curve for {model_name}", xlabel="Epochs", ylabel="Loss")
    ax.legend()
    plt.show()


def error_by_group(y_true, y_pred, groups, group_name="group"):
    """MAE and mean bias (predicted - actual) per group, e.g. Product_Category, Store_Type, Month."""
    df = pd.DataFrame({group_name: np.asarray(groups), "actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    df["abs_error"] = (df["pred"] - df["actual"]).abs()
    df["bias"] = df["pred"] - df["actual"]
    return (df.groupby(group_name)
              .agg(rows=("actual", "size"), MAE=("abs_error", "mean"), bias=("bias", "mean"))
              .sort_values("MAE", ascending=False))


def evaluate_model(y_test, y_pred, model_name, plots=True):
    """Print the metrics, optionally draw the standard plots, and return the metrics dict."""
    metrics = regression_metrics(y_test, y_pred)
    print("Model: {}".format(model_name))
    for name, value in metrics.items():
        print("{:<5}: {:.4f}".format(name.upper(), value))
    if plots:
        plot_actual_vs_predicted(y_test, y_pred, model_name)
        plot_residuals(y_test, y_pred, model_name)
    return metrics


def compare_models(results):
    """results = {model_name: metrics dict}. Returns a table sorted by RMSE (best first)."""
    table = pd.DataFrame(results).T[["mae", "mse", "rmse", "r2", "wape", "bias"]]
    return table.sort_values("rmse")