"""SmartStock AI dashboard. Call run() from app.py.

Tabs: Data explorer | Predict & explore | Inventory recommender | Model comparison
Sidebar: forecast model, service level, lead time, date window, extra inventory rules.
"""
import datetime as dt

import numpy as np
import streamlit as st

from . import data, explorer_tab, model_comparison, predict_tab, predictor, recommender as rec, recommender_tab
from .config import (DATE_COL, HIGH_DEMAND_PCT, LEAD_DAYS, MODELS, OVERSTOCK_DAYS, REVIEW_DAYS, SERVICE_LEVEL, TARGET_DAYS, TEST_FILE)

CSS = """
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 10px; padding: 12px 16px;}
</style>
"""


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def sidebar(models, y_all, test_dates):
    st.sidebar.title("📦 SmartStock AI")
    st.sidebar.caption("Demand forecasting and inventory recommendations")

    st.sidebar.subheader("Forecast model")
    scores = st.session_state.get("rmse_by_model", {})
    best = min(scores, key=scores.get) if scores else models[0].name
    names = [m.name for m in models]
    pick = st.sidebar.selectbox("Model", names, index=names.index(best),
                                format_func=lambda n: f"{n} (best RMSE)" if n == best and len(names) > 1 else n)
    spec = next(m for m in models if m.name == pick)

    st.sidebar.subheader("Inventory planning")
    level = st.sidebar.slider("Service level (%)", 90, 99, SERVICE_LEVEL,
                              help="Chance of not running out while waiting for a delivery. Higher = more safety stock.")
    by_category = data.lead_by_category()
    sources = (["Shipping_Time (from data)"] if by_category else []) + ["Fixed days"]
    source = st.sidebar.selectbox("Lead time source", sources)
    fixed = st.sidebar.number_input("Lead time (days)", 1, 30, LEAD_DAYS) if source == "Fixed days" else LEAD_DAYS
    with st.sidebar.expander("More inventory rules"):
        review = st.slider("Review period (days)", 1, 30, REVIEW_DAYS, help="Extra days of demand the stock must also cover after the lead time.")
        target = st.slider("Replenish up to (days of demand)", 7, 60, TARGET_DAYS)
        over = st.slider("Overstocking if cover is above (days)", 20, 120, OVERSTOCK_DAYS)
        pct = st.slider("High demand = top percentile of past sales", 50, 99, HIGH_DEMAND_PCT)
    rules = rec.Rules(level / 100, review, target, over, rec.high_demand_threshold(y_all, pct))
    lead = rec.LeadTime(source != "Fixed days", int(fixed), by_category)
    st.sidebar.caption(f"Reorder point = demand over (lead time + {review} days) + safety stock. High demand = at least {rules.high_demand_units:.0f} units.")

    st.sidebar.subheader("Date window")
    lo, hi = test_dates.min().date(), test_dates.max().date()
    default_lo = max(lo, hi - dt.timedelta(days=29))
    period = st.sidebar.date_input("Dates (test period)", (default_lo, hi), min_value=lo, max_value=hi)
    if not (isinstance(period, tuple) and len(period) == 2):
        period = (default_lo, hi)
    return spec, rules, lead, period


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def run():
    st.set_page_config(page_title="SmartStock AI", page_icon="📦", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("SmartStock AI")
    st.caption("Smart retail inventory prediction and optimisation")

    missing = data.missing_data_files()
    if missing:
        st.error("Data files not found:\n\n" + "\n".join(f"- `{p}`" for p in missing))
        st.code("python src/backend/pipeline.py --skip-training", language="bash")
        st.stop()
    models = predictor.available_models()
    if not models:
        st.error("No trained models found in `model/`. Train them first:")
        st.code("python src/backend/pipeline.py", language="bash")
        st.stop()

    mtime = TEST_FILE.stat().st_mtime
    meta, X, y = data.load_test(mtime)
    st.session_state["rmse_by_model"] = {m.name: rmse(y, predictor.predict_test(m, X)) for m in models}
    spec, rules, lead, period = sidebar(models, data.train_sales_distribution(), meta[DATE_COL])

    skipped = [m.name for m in MODELS if m not in models]
    if skipped:
        st.sidebar.warning("Not trained yet: " + ", ".join(skipped))

    # forecast error of the chosen model per category: the "sigma" behind safety stock and prediction ranges
    sigma, sigma_all = rec.residual_sigma(meta["Product_Category"], y, predictor.predict_test(spec, X))

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data explorer", "🔮 Predict & explore", "📦 Inventory recommender", "🧪 Model comparison"])
    with tab1:
        explorer_tab.render()
    with tab2:
        predict_tab.render(models, spec, rules, sigma, sigma_all, lead.for_category)
    with tab3:
        recommender_tab.render(meta, X, spec, rules, lead, sigma, sigma_all, period)
    with tab4:
        model_comparison.render(X, y)
