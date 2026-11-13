"""Models tab: shows what src/backend/pipeline.py trained (metrics.json, plots, reports).

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
                "Balanced accuracy": t["balanced_accuracy"], "MCC": t["mcc"], "ROC-AUC": t.get("roc_auc_ovr_macro"),
                "Understock recall": t["per_class"]["Understock"]["recall"],
                "Overstock recall": t["per_class"]["Overstock"]["recall"],
                "CV macro-F1": r["metrics"][key]["cv_f1_macro_mean"],
                "Seconds": r["metrics"][key]["seconds"],
            })
    return pd.DataFrame(rows)


def render() -> None:
    results = load_results(_signature())
    if not results:
        st.info("No trained models yet. Run `python src/backend/pipeline.py`.")
        return

    table = _summary_table(results)
    tuned = table[table["Stage"] == "Tuned"].set_index("Model")
    best = tuned["Macro-F1"].idxmax()

    # One card per model, best first. The best one is the model the dashboard uses.
    order = tuned.sort_values("Macro-F1", ascending=False).index
    for col, name in zip(st.columns(len(order)), order):
        t = tuned.loc[name]
        badge = '<span class="badge">In use</span>' if name == best else ""
        col.markdown(
            f'<div class="mcard {"best" if name == best else ""}"><div class="n">{name}{badge}</div>'
            f'<div class="f">{t["Macro-F1"]:.3f}</div>'
            f'<div class="m">macro-F1 · accuracy {t["Accuracy"]:.1%}<br>'
            f'understock found {t["Understock recall"]:.0%} · overstock found {t["Overstock recall"]:.0%}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns(2)
    with left:
        metric = st.segmented_control(
            "Compare", ["Macro-F1", "Accuracy", "MCC", "ROC-AUC"], default="Macro-F1", key="m_metric",
        ) or "Macro-F1"
        fig = px.bar(
            table, x="Model", y=metric, color="Stage", barmode="group", text_auto=".3f",
            color_discrete_map=STAGE_COLORS, category_orders={"Stage": list(STAGE_COLORS)},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        style(fig, height=330, legend=True)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", range=[0, min(1.0, float(table[metric].max()) * 1.2)])
        show(fig)
    with right:
        classes = st.segmented_control(
            "Classes", list(CLASS_COLORS), default=list(CLASS_COLORS), selection_mode="multi", key="m_classes",
        ) or list(CLASS_COLORS)
        recall = pd.DataFrame([
            {"Model": name, "Class": cls, "Recall": v["recall"]}
            for name, r in results.items()
            for cls, v in r["metrics"]["tuned"]["test"]["per_class"].items() if cls in classes
        ])
        fig = px.bar(
            recall, x="Model", y="Recall", color="Class", barmode="group", text_auto=".0%",
            color_discrete_map=CLASS_COLORS, category_orders={"Class": list(CLASS_COLORS)},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        style(fig, height=330, legend=True)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", tickformat=".0%", range=[0, 1.1])
        show(fig)

    section("Model detail")
    name = st.segmented_control("Model", list(results), default=best, key="m_detail", label_visibility="collapsed") or best
    r = results[name]
    m, folder = r["metrics"], r["folder"]
    t = m["tuned"]["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{t['accuracy']:.1%}")
    c2.metric("Macro-F1", f"{t['f1_macro']:.3f}")
    c3.metric("Understock found", f"{t['per_class']['Understock']['recall']:.0%}")
    c4.metric("Overstock found", f"{t['per_class']['Overstock']['recall']:.0%}")

    a, b = st.columns([2, 3])
    with a:
        confusion = folder / "confusion_matrix.png"
        if confusion.exists():
            st.image(str(confusion), caption="Confusion matrix (rows sum to 100%)")
    with b:
        plots = [p for p in sorted(folder.glob("*.png")) if p.name != "confusion_matrix.png"]
        if plots:
            picked = st.segmented_control(
                "Drivers", [p.name for p in plots], default=plots[0].name, key="m_plot",
                format_func=lambda n: FEATURE_PLOT_TITLES.get(n, n.removesuffix(".png").replace("_", " ")).split(" (")[0],
            ) or plots[0].name
            if (folder / picked).exists():
                st.image(str(folder / picked))

    with st.expander("Settings, split and full results"):
        params = m["tuned"].get("best_params", {})
        if params:
            st.dataframe(
                pd.DataFrame({"Setting": [k.replace("clf__", "") for k in params], "Value": [str(v) for v in params.values()]}),
                hide_index=True,
            )
        st.caption(f"{m['n_train']:,} training rows, {m['n_test']:,} test rows. " + m["split"].replace("\n", " | "))
        shown = table.sort_values(["Stage", "Macro-F1"], ascending=[False, False])
        st.dataframe(
            styled(shown, {c: "{:.3f}" for c in ["Accuracy", "Macro-F1", "Balanced accuracy", "MCC", "ROC-AUC",
                                                 "Understock recall", "Overstock recall", "CV macro-F1"]} | {"Seconds": "{:,.0f}"}),
            hide_index=True, height=35 * (len(shown) + 1) + 3,
        )
        report = folder / "classification_report.txt"
        if report.exists():
            st.code(report.read_text(), language=None)
