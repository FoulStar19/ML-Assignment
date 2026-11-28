import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve, validation_curve

LINE = "─" * 45

# ─────────────────────────────────────────────
# 1. EVALUATION METRICS (MAE, MSE, RMSE, R², MAPE, Bias)
# ─────────────────────────────────────────────
def evaluate_model(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE and handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = np.nan_to_num(mape)  # Replace NaN with 0

    # Calculate Bias
    bias = np.mean(y_pred - y_true)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "MAPE": mape,
        "Bias": bias
    }

def print_evaluation_metrics(metrics):
    print(LINE)
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R² Score: {metrics['R²']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    print(f"Bias: {metrics['Bias']:.4f}")
    print(LINE + "\n")

def default_vs_tuned_model_comparison(y_true, y_pred_default, y_pred_tuned):
    # Evaluate default model
    metrics_default = evaluate_model(y_true, y_pred_default)
    
    # Evaluate tuned model
    metrics_tuned = evaluate_model(y_true, y_pred_tuned)
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Metric': list(metrics_default.keys()),
        'Default Model': list(metrics_default.values()),
        'Tuned Model': list(metrics_tuned.values())
    })
    
    return comparison_df

def compare_model_results(base_results, tuned_results, split="test_metrics"):
    # base_results / tuned_results are the dicts returned by the train_*_base / train_*_tuned functions
    # Rows = metrics, columns = each model's default and tuned versions
    data = {}
    for model_name in base_results:
        data[f"{model_name} (Default)"] = base_results[model_name][split]
        data[f"{model_name} (Tuned)"] = tuned_results[model_name][split]
    comparison_df = pd.DataFrame(data)
    comparison_df.index.name = "Metric"
    print(comparison_df.to_string())
    return comparison_df

# ─────────────────────────────────────────────
# 2. GRAPH PLOTS
# ─────────────────────────────────────────────
def actual_vs_predicted_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Values')
    return fig, ax

def residuals_plot(y_true, y_pred):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted Values')
    return fig, ax

def feature_importance_plot(feature_importances, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(feature_importances)[::-1]
    ax.bar(range(len(feature_importances)), feature_importances[indices], align='center')
    ax.set_xticks(range(len(feature_importances)))
    ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
    ax.set_title('Feature Importances')
    ax.set_ylabel('Importance')
    return fig, ax

def permutation_importance_plot(perm_importances, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(perm_importances)[::-1]
    ax.bar(range(len(perm_importances)), perm_importances[indices], align='center')
    ax.set_xticks(range(len(perm_importances)))
    ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
    ax.set_title('Permutation Importances')
    ax.set_ylabel('Importance')
    return fig, ax

def plot_learning_curve(train_sizes, train_scores, test_scores):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='r')
    ax.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                    np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1, color='g')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    return fig, ax

def training_validation_curve_plot(param_range, train_scores, val_scores, param_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_range, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    ax.plot(param_range, np.mean(val_scores, axis=1), 'o-', color='g', label='Validation score')
    ax.fill_between(param_range, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='r')
    ax.fill_between(param_range, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                    np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='g')
    if min(param_range) > 0 and max(param_range) / min(param_range) >= 100:
        ax.set_xscale('log')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(f'Training and Validation Curve for {param_name}')
    ax.legend(loc='best')
    return fig, ax

def loss_curve_plot(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_losses, label='Training Loss', color='r')
    if val_losses is not None:
        ax.plot(val_losses, label='Validation Loss', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend(loc='best')
    return fig, ax

def confusion_matrix_plot(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Annotate the cells with counts
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
    
    return fig, ax

# ─────────────────────────────────────────────
# 3. SHARED HELPER FUNCTIONS
# ─────────────────────────────────────────────
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)

def save_model(model, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def save_results(results, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def generate_model_plots(model, model_name, X_train, y_train, X_test, y_test, output_dir,
                         feature_names=None, importances=None, loss_curves=None,
                         validation_param=None):
    
    matplotlib.use("Agg")  # save to files only, no GUI windows
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(np.asarray(X_train).shape[1])]
    feature_names = np.array(feature_names)

    def top_n(values, n=20):
        values = np.asarray(values)
        idx = np.argsort(values)[::-1][:n]
        return values[idx], feature_names[idx]

    def make(suffix, build):
        try:
            fig, _ = build()
            path = os.path.join(output_dir, f"{model_name}_{suffix}.png")
            save_plot(fig, path)
            print(f"Plot saved to {path}")
        except Exception as e:
            plt.close("all")
            print(f"[WARN] Could not create {suffix} plot for {model_name}: {e}")

    make("actual_vs_predicted", lambda: actual_vs_predicted_plot(y_true, y_pred))
    make("residuals", lambda: residuals_plot(y_true, y_pred))

    if importances is not None:
        make("feature_importance", lambda: feature_importance_plot(*top_n(importances)))

    def perm_plot():
        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42,
                                      scoring="neg_mean_squared_error", n_jobs=1)
        return permutation_importance_plot(*top_n(perm.importances_mean))
    make("permutation_importance", perm_plot)

    def lc_plot():
        sizes, tr, te = learning_curve(clone(model), X_train, y_train, cv=3, scoring="r2",
                                       train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1)
        return plot_learning_curve(sizes, tr, te)
    make("learning_curve", lc_plot)

    if validation_param is not None:
        name, values = validation_param
        def vc_plot():
            tr, te = validation_curve(clone(model), X_train, y_train, param_name=name,
                                      param_range=values, cv=3, scoring="r2", n_jobs=1)
            return training_validation_curve_plot(values, tr, te, name)
        make(f"validation_curve_{name}", vc_plot)

    if loss_curves is not None:
        make("loss_curve", lambda: loss_curve_plot(*loss_curves))