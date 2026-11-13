"""
compare_models.py - collects the results of all four models into one comparison.

Run AFTER the four model scripts:   python compare_models.py
Outputs (in models/comparison/): comparison_table.csv, comparison_macro_f1.png,
                                  comparison_class_recall.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
OUT_DIR = MODELS_DIR / "comparison"
BASELINE_KEY = "Logistic Regression"          # first match is treated as the baseline


def main():
    files = sorted(MODELS_DIR.glob("*/outputs/metrics.json"))
    if not files:
        raise SystemExit("No metrics.json found - run the four model scripts first.")
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for f in files:
        r = json.loads(f.read_text())
        for stage in ("default", "tuned"):
            t = r[stage]["test"]
            rows.append({
                "model": r["model"], "stage": stage,
                "cv_f1_macro": r[stage]["cv_f1_macro_mean"],
                "test_accuracy": t["accuracy"], "test_balanced_acc": t["balanced_accuracy"],
                "test_precision_macro": t["precision_macro"], "test_recall_macro": t["recall_macro"],
                "test_f1_macro": t["f1_macro"], "test_mcc": t["mcc"],
                "test_roc_auc": t.get("roc_auc_ovr_macro", np.nan),
                "recall_understock": t["per_class"]["Understock"]["recall"],
                "recall_overstock": t["per_class"]["Overstock"]["recall"],
            })
    table = pd.DataFrame(rows)

    # improvement over the baseline, compared within the same stage (default vs default, tuned vs tuned)
    is_base = table.model.str.contains(BASELINE_KEY)
    base_f1 = table[is_base].set_index("stage").test_f1_macro
    table["f1_gain_vs_baseline"] = table.test_f1_macro - table.stage.map(base_f1)
    table.round(4).to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print(table.round(4).to_string(index=False))

    tuned = table[table.stage == "tuned"].set_index("model")
    default = table[table.stage == "default"].set_index("model")

    # chart 1: macro-F1 default vs tuned
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tuned))
    ax.bar(x - 0.2, default.loc[tuned.index, "test_f1_macro"], 0.4, label="default")
    ax.bar(x + 0.2, tuned.test_f1_macro, 0.4, label="tuned")
    ax.set_xticks(x, tuned.index, rotation=15)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Model comparison - macro-F1 on future test data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_macro_f1.png", dpi=150)
    plt.close(fig)

    # chart 2: recall on the two classes that matter for the business
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, tuned.recall_understock, 0.4, label="Understock recall")
    ax.bar(x + 0.2, tuned.recall_overstock, 0.4, label="Overstock recall")
    ax.set_xticks(x, tuned.index, rotation=15)
    ax.set_ylabel("recall")
    ax.set_title("How many under/overstock cases does each model catch? (tuned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_class_recall.png", dpi=150)
    plt.close(fig)

    best = tuned.test_f1_macro.idxmax()
    print(f"\nBest tuned model by macro-F1: {best} ({tuned.test_f1_macro.max():.4f})")


if __name__ == "__main__":
    main()
