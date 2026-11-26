import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score)

def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr,tpr,'k--')
    ax.plot([0,1],[0,1],'k--',linewidth=.5)
    ax.grid(True)
    ax.set(title = 'ROC Curve for {}'.format(model_name), xlabel = 'False Positive Rate', ylabel = 'True Positive Rate', xlim = [-0.01, 1.01], ylim = [-0.01, 1.01])
    plt.show()

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set(title = 'Confusion Matrix for {}'.format(model_name), xlabel = 'Predicted', ylabel = 'Actual')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred, model_name):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(recall, precision, color='b')
    ax.set(title = 'Precision-Recall Curve for {}'.format(model_name), xlabel = 'Recall', ylabel = 'Precision', xlim = [-0.01, 1.01], ylim = [-0.01, 1.01])
    plt.show()

def plot_loss_curve(train_losses, val_losses, model_name):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set(title = 'Loss Curve for {}'.format(model_name), xlabel = 'Epochs', ylabel = 'Loss')
    ax.legend()
    plt.show()

def plot_accuracy_curve(train_accuracies, val_accuracies, model_name):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(train_accuracies, label='Train Accuracy')
    ax.plot(val_accuracies, label='Validation Accuracy')
    ax.set(title = 'Accuracy Curve for {}'.format(model_name), xlabel = 'Epochs', ylabel = 'Accuracy')
    ax.legend()
    plt.show()

def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Model: {}".format(model_name))
    print("Accuracy: {:.4f}".format(accuracy))
    print("ROC AUC: {:.4f}".format(roc_auc))
    plot_roc(y_test, y_pred, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_precision_recall_curve(y_test, y_pred, model_name)