"""Dashboard page: sidebar settings, the planning context and the tab layout.

Run from the project root:   streamlit run app.py
"""
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import model_results
import tabs
from config import PROCESSED_FILE, RAW_FILE, REQUIRED_COLUMNS, ROOT_DIR
from planning import apply_policy, demand_stats, load_data
from predictor import (FEATURES, add_predictions, best_model_info, demand_model_info, load_best_model,
                       with_model_demand)
from recommender import plan_transfers, recommend
from theme import brand, hero, inject_css

from data_transforming import OVERSTOCK_COVER_DAYS  # noqa: E402  (the label the models were trained on; src/backend is on the path)


# ─────────────────────────────────────────────
# 1. DATA CHECK
# ─────────────────────────────────────────────
def load_base() -> pd.DataFrame:
    """Processed data (built from the raw file on first start), or a clear message and a stop."""
    if not PROCESSED_FILE.exists() and RAW_FILE.exists():
        from pipeline import run_pipeline          # src/backend is on the path (see predictor.py)
        with st.spinner("Building the processed data (first start only)..."):
            run_pipeline(train_models=False)
    base = load_data()
    if base.empty:
        st.error(f"No data found. Add `{RAW_FILE.relative_to(ROOT_DIR)}`, then run `python src/backend/pipeline.py`.")
        st.stop()
    missing = [c for c in REQUIRED_COLUMNS + FEATURES + ["stock_status", "stock_cover"] if c not in base.columns]
    if missing:
        st.error(f"The processed data is out of date (missing {', '.join(missing)}). Run `python src/backend/pipeline.py`.")
        st.stop()
    return base


# ─────────────────────────────────────────────
# 2. SIDEBAR  ->  Settings
# ─────────────────────────────────────────────
@dataclass
class Settings:
    lead: int
    review: int
    service: float
    window: int
    over_days: float
    watch_days: float
    use_transfers: bool
    stores: list
    start: pd.Timestamp
    end: pd.Timestamp
    use_demand_model: bool


PRESETS = {                      # plan style -> (lead time, order every, service level, review window)
    "Lean": (3, 1, 0.90, 14),
    "Balanced": (4, 1, 0.95, 14),
    "Safe": (6, 2, 0.99, 14),
}


def _apply_preset() -> None:
    choice = st.session_state.get("preset")
    if choice in PRESETS:
        st.session_state["lead"], st.session_state["review"], st.session_state["service"], st.session_state["window"] = PRESETS[choice]


def _clear_preset() -> None:
    st.session_state["preset"] = None            # a slider was moved by hand, so no preset describes the settings any more


def render_sidebar(base: pd.DataFrame, has_demand_model: bool) -> Settings:
    for key, value in (("preset", "Balanced"), ("lead", 4), ("review", 1), ("service", 0.95), ("window", 14)):
        st.session_state.setdefault(key, value)
    with st.sidebar:
        brand()
        with st.expander("Scope", expanded=True):
            all_stores = sorted(base["store_id"].unique())
            stores = st.multiselect("Stores", all_stores, default=all_stores)
            min_d, max_d = base["date"].min().date(), base["date"].max().date()
            picked = st.date_input("Dates", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        with st.expander("Ordering", expanded=True):
            st.segmented_control("Plan style", list(PRESETS), key="preset", on_change=_apply_preset)
            lead = st.slider("Lead time", 1, 10, key="lead", format="%d d", on_change=_clear_preset)
            review = st.slider("Order every", 1, 7, key="review", format="%d d", on_change=_clear_preset)
            service = st.select_slider("Service level", options=[0.90, 0.95, 0.975, 0.99], key="service",
                                       format_func=lambda v: f"{v:.1%}", on_change=_clear_preset)
            window = st.slider("Review window", 7, 60, key="window", format="%d d", on_change=_clear_preset)

        with st.expander("More", expanded=False):
            watch_days = st.slider("Reorder-soon window", 0.5, 3.0, 0.5, step=0.5, format="%.1f d")
            use_transfers = st.toggle("Store transfers", value=True)
            use_demand = st.toggle("XGBoost demand", value=True) if has_demand_model else False

    if not stores:
        st.warning("Select at least one store.")
        st.stop()
    if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
        st.info("Pick an end date.")
        st.stop()
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

    return Settings(
        lead=lead, review=review, service=service, window=window, over_days=OVERSTOCK_COVER_DAYS,
        watch_days=watch_days, use_transfers=use_transfers, stores=stores, start=start, end=end,
        use_demand_model=use_demand,
    )


# ─────────────────────────────────────────────
# 3. PLANNING CONTEXT  ->  Context (shared by every tab)
# ─────────────────────────────────────────────
@dataclass
class Context:
    plan: pd.DataFrame            # every record with its policy numbers
    lines: pd.DataFrame           # the latest record of each store-product in the review window
    hist: pd.DataFrame            # the whole scope (Trends tab)
    transfers: pd.DataFrame
    recs: pd.DataFrame
    asof_date: pd.Timestamp
    lead: int
    model: dict
    demand: dict | None           # XGBoost demand model in use (None = subcategory averages)
    over_days: float
    use_transfers: bool
    n_lines: int
    n_critical: int
    n_under: int
    n_over: int
    n_healthy: int
    net_units: float
    net_value: float
    moved_units: int
    moved_value: float
    excess_value: float


def build_context(base: pd.DataFrame, s: Settings, model, model_info: dict, demand: dict | None) -> Context:
    lead, review, service, window = s.lead, s.review, s.service, s.window
    over_days, watch_days, use_transfers = s.over_days, s.watch_days, s.use_transfers
    stores, start, end = s.stores, s.start, s.end

    expected = with_model_demand(demand) if demand is not None else demand_stats()   # XGBoost sets the expected demand
    plan = apply_policy(expected, lead, review, service, over_days)
    scope = plan[plan["store_id"].isin(stores) & plan["date"].between(start, end)]
    if scope.empty:
        st.warning("No records for that range.")
        st.stop()

    asof_date = scope["date"].max()
    recent = scope[scope["date"] > asof_date - pd.Timedelta(days=window)]        # the review window ending on the last date
    recent = recent.sort_values(["date", "transaction_id"]).drop_duplicates(["store_id", "product_id"], keep="last")
    lines = add_predictions(recent, model, review)                               # the best model decides the status
    hist = scope
    transfers = plan_transfers(lines) if use_transfers else plan_transfers(lines.iloc[0:0])
    recs = recommend(lines, transfers, lead, over_days, watch_days)

    n_lines = len(lines)
    n_critical = int((recs["priority"] == "Critical").sum())
    n_under = int((recs["status"] == "Understock risk").sum())
    n_over = int((recs["status"] == "Overstock").sum())
    n_healthy = n_lines - n_under - n_over
    net_units = float(recs["net_order"].sum())
    net_value = float(recs["net_order_value"].sum())
    moved_units = int(transfers["qty"].sum()) if not transfers.empty else 0
    moved_value = float(transfers["value"].sum()) if not transfers.empty else 0.0
    excess_value = float(recs["excess_value"].sum())

    return Context(
        plan=plan, lines=lines, hist=hist, transfers=transfers, recs=recs, asof_date=asof_date,
        lead=lead, model=model_info, demand=demand, over_days=over_days, use_transfers=use_transfers, n_lines=n_lines, n_critical=n_critical,
        n_under=n_under, n_over=n_over, n_healthy=n_healthy, net_units=net_units, net_value=net_value,
        moved_units=moved_units, moved_value=moved_value, excess_value=excess_value,
    )


# ─────────────────────────────────────────────
# 4. PAGE
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Smart Inventory Replenishment", page_icon="📦", layout="wide")
    inject_css()

    base = load_base()
    info = best_model_info()
    if info is None:
        st.error("No trained model found. Run `python src/backend/pipeline.py`.")
        st.stop()
    model = load_best_model(info)
    demand_info = demand_model_info()

    s = render_sidebar(base, demand_info is not None)
    ctx = build_context(base, s, model, info, demand_info if s.use_demand_model else None)

    models = [info["name"]] + (["XGBoost demand"] if ctx.demand else [])
    hero(f"{ctx.asof_date:%d %b %Y}", models, ctx.n_under, ctx.n_healthy, ctx.n_over)

    tab_over, tab_rec, tab_trends, tab_models = st.tabs(["Overview", "Actions", "Trends", "Models"])
    with tab_over:
        tabs.tab_overview(ctx)
    with tab_rec:
        tabs.tab_recommendations(ctx)
    with tab_trends:
        tabs.tab_trends(ctx)
    with tab_models:
        model_results.render()
