import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib.resources import path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R² Score: {metrics['R²']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    print(f"Bias: {metrics['Bias']:.4f}")

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
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(f'Training and Validation Curve for {param_name}')
    ax.legend(loc='best')
    return fig, ax

def loss_curve_plot(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_losses, label='Training Loss', color='r')
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