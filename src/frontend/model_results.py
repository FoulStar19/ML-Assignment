"""Model results tab: shows what src/backend/pipeline.py trained (metrics.json, plots, reports).

Nothing is computed here. Each model folder under model/<name>/outputs/ is read as it is,
so the tab updates the next time the pipeline is run.
"""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from config import BLUE, MODEL_DIR, STATUS_COLORS
from theme import section, show, style, styled

CLASS_COLORS = {
    "Understock": STATUS_COLORS["Understock risk"],
    "Healthy": STATUS_COLORS["Healthy"],
    "Overstock": STATUS_COLORS["Overstock"],
}
STAGE_COLORS = {"Default settings": "#A9B5C1", "Tuned": BLUE}
FEATURE_PLOT_TITLES = {                       # file name -> caption (anything else uses its file name)
    "coefficients_understock.png": "Top drivers of Understock (coefficients)",
    "coefficients_overstock.png": "Top drivers of Overstock (coefficients)",
    "impurity_importance.png": "Impurity importance",
    "gain_importance.png": "Gain importance",
    "permutation_importance.png": "Permutation importance (drop in test macro-F1)",
    "training_curves.png": "Training curves",
}


@st.cache_data(show_spinner=False)
def load_results(signature: tuple) -> dict:
    """{model name: {"folder": Path, "metrics": dict}} for every model that has a metrics.json.

    `signature` (file modification times) only exists to refresh the cache after a new training run.
    """
    found = {}
    for path in sorted(MODEL_DIR.glob("*/outputs/metrics.json")):
        metrics = json.loads(path.read_text())
        found[metrics["model"]] = {"folder": path.parent, "metrics": metrics}
    return found


def _signature() -> tuple:
    return tuple((str(p), p.stat().st_mtime) for p in sorted(MODEL_DIR.glob("*/outputs/metrics.json")))


def _summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        for stage, key in (("Default settings", "default"), ("Tuned", "tuned")):
            t = r["metrics"][key]["test"]
            rows.append({
                "Model": name, "Stage": stage,
                "Accuracy": t["accuracy"], "Macro-F1": t["f1_macro"],
                "Balanced accuracy": t["balanced_accuracy"], "MCC": t["mcc"],
                "Understock recall": t["per_class"]["Understock"]["recall"],
                "Overstock recall": t["per_class"]["Overstock"]["recall"],
                "CV macro-F1": r["metrics"][key]["cv_f1_macro_mean"],
                "Seconds": r["metrics"][key]["seconds"],
            })
    return pd.DataFrame(rows)


def render() -> None:
    results = load_results(_signature())
    if not results:
        st.info(
            "No trained models found yet. Run `python src/backend/pipeline.py` to train the four models. "
            "Their results will appear here."
        )
        return

    table = _summary_table(results)
    tuned = table[table["Stage"] == "Tuned"].set_index("Model")
    best = tuned["Macro-F1"].idxmax()
    best_under = tuned["Understock recall"].idxmax()
    best_over = tuned["Overstock recall"].idxmax()
    any_metrics = next(iter(results.values()))["metrics"]

    st.markdown(
        f'<div class="lede">{best} has the best balanced score (macro-F1 {tuned.loc[best, "Macro-F1"]:.3f}) '
        f'on the held-out test period.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="lede-sub">{best_under} catches the most understocked lines '
        f'({tuned.loc[best_under, "Understock recall"]:.0%} recall) and {best_over} the most overstocked '
        f'({tuned.loc[best_over, "Overstock recall"]:.0%}). Each model predicts the stock status over the '
        f'next 7 days and is tuned with {any_metrics["n_iter"]} random-search candidates and '
        f'{any_metrics["cv_splits"]}-fold time-series cross-validation.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            table, x="Model", y="Macro-F1", color="Stage", barmode="group", text_auto=".3f",
            color_discrete_map=STAGE_COLORS, category_orders={"Stage": list(STAGE_COLORS)},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        style(fig, height=340, title="Macro-F1 on the test set", legend=True)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", range=[0, min(1.0, float(table["Macro-F1"].max()) * 1.2)])
        show(fig)
    with right:
        recall = pd.DataFrame([
            {"Model": name, "Class": cls, "Recall": v["recall"]}
            for name, r in results.items()
            for cls, v in r["metrics"]["tuned"]["test"]["per_class"].items()
        ])
        fig = px.bar(
            recall, x="Model", y="Recall", color="Class", barmode="group", text_auto=".0%",
            color_discrete_map=CLASS_COLORS, category_orders={"Class": list(CLASS_COLORS)},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        style(fig, height=340, title="Share of each class found (recall, tuned models)", legend=True)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", tickformat=".0%", range=[0, 1.1])
        show(fig)

    section("All results", "Test-set scores for every model, before and after tuning.")
    shown = table.sort_values(["Stage", "Macro-F1"], ascending=[False, False])
    st.dataframe(
        styled(shown, {
            "Accuracy": "{:.3f}", "Macro-F1": "{:.3f}", "Balanced accuracy": "{:.3f}", "MCC": "{:.3f}",
            "Understock recall": "{:.3f}", "Overstock recall": "{:.3f}", "CV macro-F1": "{:.3f}",
            "Seconds": "{:,.0f}",
        }),
        hide_index=True, height=35 * (len(shown) + 1) + 3,
    )
    st.caption(
        "Macro-F1 averages the three classes equally, so it is not dominated by the common Healthy class. "
        "Recall is the share of truly understocked (or overstocked) lines the model flagged."
    )

    section("Model detail")
    name = st.selectbox("Model", list(results), index=list(results).index(best))
    r = results[name]
    m, folder = r["metrics"], r["folder"]

    c1, c2, c3, c4 = st.columns(4)
    t = m["tuned"]["test"]
    c1.metric("Accuracy", f"{t['accuracy']:.1%}")
    c2.metric("Macro-F1", f"{t['f1_macro']:.3f}")
    c3.metric("Understock recall", f"{t['per_class']['Understock']['recall']:.0%}")
    c4.metric("Overstock recall", f"{t['per_class']['Overstock']['recall']:.0%}")

    a, b = st.columns([2, 3])
    with a:
        confusion = folder / "confusion_matrix.png"
        if confusion.exists():
            st.image(str(confusion), caption="Confusion matrix (each row sums to 100%)")
    with b:
        st.markdown(f"**Data split** \n{m['n_train']:,} training rows, {m['n_test']:,} test rows.  \n"
                    + m["split"].replace("\n", "  \n"))
        params = m["tuned"].get("best_params", {})
        if params:
            st.markdown("**Best settings found by tuning**")
            st.dataframe(
                pd.DataFrame({"Setting": [k.replace("clf__", "") for k in params],
                              "Value": [str(v) for v in params.values()]}),
                hide_index=True,
            )
        report = folder / "classification_report.txt"
        if report.exists():
            with st.expander("Classification report"):
                st.code(report.read_text(), language=None)

    plots = [p for p in sorted(folder.glob("*.png")) if p.name != "confusion_matrix.png"]
    if plots:
        section("What drives the predictions")
        cols = st.columns(2)
        for i, p in enumerate(plots):
            cols[i % 2].image(str(p), caption=FEATURE_PLOT_TITLES.get(p.name, p.stem.replace("_", " ")))

    with st.expander("How the models are trained and judged"):
        st.markdown(
            "Rows are split by date: the most recent 20% of days are held out as the test period, with a 7-day gap "
            "so no label peeks into the test window. Tuning uses time-series cross-validation, so each fold "
            "trains on the past and validates on the block after it. The label is the stock status over the next "
            "7 days, based on forward days of cover. Understock is below 1 day, overstock above 3 days."
        )
