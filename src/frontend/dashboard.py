"""Dashboard page: sidebar settings, the planning context and the tab layout.

Run from the project root:   streamlit run app.py
"""
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import model_results
import tabs
from config import MIN_HISTORY_DAYS, PROCESSED_FILE, RAW_FILE, REQUIRED_COLUMNS, ROOT_DIR
from planning import apply_policy, demand_stats, load_data
from predictor import FEATURES, add_predictions, best_model_info, load_best_model
from recommender import plan_transfers, recommend
from theme import inject_css

OVERSTOCK_COVER_DAYS = 3.0   # the label the models were trained on (see src/backend/data_transforming.py)


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
    missing = [c for c in REQUIRED_COLUMNS + FEATURES + ["stock_status", "forward_cover_days"] if c not in base.columns]
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
    lookback: int
    over_days: float
    watch_days: float
    use_transfers: bool
    stores: list
    start: pd.Timestamp
    end: pd.Timestamp
    cats: list
    regions: list


def render_sidebar(base: pd.DataFrame) -> Settings:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="name">📦 Inventory Control</div>'
            '<div class="tag">Replenishment planner</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Scope", expanded=True):
            all_stores = sorted(base["store_id"].unique())
            stores = st.multiselect("Stores", all_stores, default=all_stores)
            min_d, max_d = base["date"].min().date(), base["date"].max().date()
            picked = st.date_input("Dates", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                                   help="The plan is made for the last day.")

        with st.expander("Ordering", expanded=True):
            lead = st.slider("Lead time (days)", 1, 7, 1)
            review = st.slider("Days between orders", 1, 7, 1)
            service = st.select_slider(
                "Service level", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
                format_func=lambda v: f"{v:.1%}", help="Higher means more safety stock.",
            )
            lookback = st.slider("Demand look-back (days)", 14, 90, 30)

        with st.expander("More", expanded=False):
            watch_days = st.slider("Reorder-soon window (days)", 0.5, 3.0, 0.5, step=0.5)
            use_transfers = st.toggle("Store-to-store transfers", value=True)
            all_cats = sorted(base["category"].unique())
            all_regions = sorted(base["region"].unique())
            cats = st.multiselect("Category (Trends)", all_cats, default=all_cats)
            regions = st.multiselect("Region (Trends)", all_regions, default=all_regions)

    if not stores:
        st.warning("Select at least one store in the sidebar.")
        st.stop()
    if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
        st.info("Pick an end date to finish the date range.")
        st.stop()
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

    return Settings(
        lead=lead, review=review, service=service, lookback=lookback, over_days=OVERSTOCK_COVER_DAYS,
        watch_days=watch_days, use_transfers=use_transfers, stores=stores, start=start, end=end,
        cats=cats, regions=regions,
    )


# ─────────────────────────────────────────────
# 3. PLANNING CONTEXT  ->  Context (shared by every tab)
# ─────────────────────────────────────────────
@dataclass
class Context:
    plan: pd.DataFrame            # every store-product-day with its policy numbers
    lines: pd.DataFrame           # the plan date only, one row per store-product line
    hist: pd.DataFrame            # scope after the history filters (Demand and Stock health tabs)
    transfers: pd.DataFrame
    recs: pd.DataFrame
    asof_date: pd.Timestamp
    lead: int
    model: dict
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


def build_context(base: pd.DataFrame, s: Settings, model, model_info: dict) -> Context:
    lead, review, service, lookback = s.lead, s.review, s.service, s.lookback
    over_days, watch_days, use_transfers = s.over_days, s.watch_days, s.use_transfers
    stores, start, end, cats, regions = s.stores, s.start, s.end, s.cats, s.regions

    plan = apply_policy(demand_stats(lookback), lead, review, service, over_days)
    scope = plan[plan["store_id"].isin(stores) & plan["date"].between(start, end)]
    if scope.empty:
        st.warning(
            f"No plan is available for that range. Each line needs {MIN_HISTORY_DAYS} days of history first, "
            f"so plans start on {(base['date'].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)).date()}."
        )
        st.stop()

    asof_date = scope["date"].max()
    lines = add_predictions(scope[scope["date"] == asof_date], model, review)   # the best model decides the status
    hist = scope[scope["category"].isin(cats) & scope["region"].isin(regions)]
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
        lead=lead, model=model_info, over_days=over_days, use_transfers=use_transfers, n_lines=n_lines, n_critical=n_critical,
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
        st.error("No trained model found. Run `python src/backend/pipeline.py` to train the models.")
        st.stop()
    model = load_best_model(info)

    s = render_sidebar(base)
    ctx = build_context(base, s, model, info)

    needs_action = int((ctx.recs["priority"] != "None").sum())
    st.markdown(
        f"""
        <div class="banner">
          <div>
            <h1>Smart Inventory Replenishment</h1>
            <div class="sub">{needs_action} of {ctx.n_lines} lines need action · {len(s.stores)} stores</div>
          </div>
          <div class="right">
            <div class="pill"><div class="l">Plan date</div><div class="v">{ctx.asof_date:%d %b %Y}</div></div>
            <div class="pill"><div class="l">Model in use</div><div class="v">{info["name"]} · F1 {info["f1"]:.3f}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_over, tab_rec, tab_trends, tab_models = st.tabs(["Overview", "Actions", "Trends", "Models"])
    with tab_over:
        tabs.tab_overview(ctx)
    with tab_rec:
        tabs.tab_recommendations(ctx)
    with tab_trends:
        tabs.tab_trends(ctx)
    with tab_models:
        model_results.render()
