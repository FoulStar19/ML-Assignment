"""SmartStock AI dashboard. Call run() from app.py.

Tabs: Inventory Overview | Predict & Recommend | Model Comparison
Sidebar: forecast model, inventory rules, date window.
"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from . import charts, data, predictor, recommender as rec
from .config import (MODELS, TEST_FILE, DATE_COL, FEATURES, HIGH_DEMAND_PCT, OVERSTOCK_DAYS, STATUS_COLORS, STATUS_ICONS, TARGET,
                     TARGET_DAYS, UNDERSTOCK_DAYS)

try:                                   # newer Streamlit uses width="stretch"; older versions use use_container_width
    import inspect
    STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.altair_chart).parameters else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}

CSS = """
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 10px; padding: 12px 16px;}
</style>
"""


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

    st.sidebar.subheader("Inventory rules")
    under = st.sidebar.slider("Understocking if cover is below (days)", 3, 30, UNDERSTOCK_DAYS)
    over = st.sidebar.slider("Overstocking if cover is above (days)", under + 5, 120, max(OVERSTOCK_DAYS, under + 5))
    target = st.sidebar.slider("Replenish / trim to (days of cover)", under, over, min(max(TARGET_DAYS, under), over))
    pct = st.sidebar.slider("High demand = top percentile of past sales", 50, 99, HIGH_DEMAND_PCT)
    rules = rec.Rules(under, over, target, rec.high_demand_threshold(y_all, pct))
    st.sidebar.caption(f"Days of cover = stock ÷ predicted daily units. High demand = at least {rules.high_demand_units:.0f} units.")

    st.sidebar.subheader("Overview period")
    lo, hi = test_dates.min().date(), test_dates.max().date()
    default_lo = max(lo, hi - dt.timedelta(days=29))
    period = st.sidebar.date_input("Dates (test period)", (default_lo, hi), min_value=lo, max_value=hi)
    if not (isinstance(period, tuple) and len(period) == 2):
        period = (default_lo, hi)
    return spec, rules, period


# ─────────────────────────────────────────────
# Tab 1: Inventory overview
# ─────────────────────────────────────────────
def overview_tab(meta, X, spec, rules, period):
    pred = predictor.predict_test(spec, X)
    table = pd.concat([meta.reset_index(drop=True), pd.Series(pred, name="Predicted_Units")], axis=1)
    table = pd.concat([table, rec.assess(pred, table["Inventory_Level"], rules)], axis=1)
    table = table[(table[DATE_COL].dt.date >= period[0]) & (table[DATE_COL].dt.date <= period[1])]

    f1, f2, f3 = st.columns(3)
    stores = f1.multiselect("Store", sorted(table["Store_ID"].unique()), placeholder="All stores")
    cats = f2.multiselect("Category", sorted(table["Product_Category"].dropna().unique()), placeholder="All categories")
    statuses = f3.multiselect("Inventory status", list(STATUS_COLORS), placeholder="All statuses")
    if stores:
        table = table[table["Store_ID"].isin(stores)]
    if cats:
        table = table[table["Product_Category"].isin(cats)]
    if statuses:
        table = table[table["Status"].isin(statuses)]

    if table.empty:
        st.info("No records match these filters.")
        return

    c = st.columns(5)
    c[0].metric("Records", f"{len(table):,}")
    for col, s in zip(c[1:4], STATUS_COLORS):
        n = int((table["Status"] == s).sum())
        col.metric(f"{STATUS_ICONS[s]} {s}", f"{n:,}", f"{n / len(table):.0%} of records", delta_color="off")
    c[4].metric("⭐ High demand", f"{int(table['High_Demand'].sum()):,}", f"{table['High_Demand'].mean():.0%} of records", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Records by status")
        st.altair_chart(charts.status_bar(table), **STRETCH)
        st.subheader("Days of cover")
        st.altair_chart(charts.cover_histogram(table, rules), **STRETCH)
        st.caption("Dashed lines mark the understocking and overstocking limits.")
    with right:
        st.subheader("Status by category")
        st.altair_chart(charts.status_by_category(table), **STRETCH)

    st.subheader("Priority ranking")
    st.caption("Highest priority first: low cover (understocking) and large excess (overstocking) rank above sufficient stock; high-demand items get a bonus.")
    show = table.sort_values(["Priority", "Cover_Days"], ascending=[False, True]).head(500).copy()
    show["Status"] = show["Status"].map(rec.label)
    show[DATE_COL] = show[DATE_COL].dt.date
    cols = [DATE_COL, "Store_ID", "Product_ID", "Product_Category", "Product_Subcategory", "Inventory_Level",
            "Predicted_Units", "Cover_Days", "Status", "High_Demand", "Priority", "Action"]
    st.dataframe(show[cols], hide_index=True, **STRETCH, height=420, column_config={
        "Predicted_Units": st.column_config.NumberColumn("Predicted units", format="%.1f"),
        "Inventory_Level": st.column_config.NumberColumn("Stock"),
        "Cover_Days": st.column_config.NumberColumn("Cover (days)", format="%.1f"),
        "High_Demand": st.column_config.CheckboxColumn("High demand"),
        "Priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.0f"),
        "Product_Category": "Category", "Product_Subcategory": "Subcategory", "Store_ID": "Store", "Product_ID": "Product"})
    st.download_button("Download this view (CSV)", table.sort_values("Priority", ascending=False).to_csv(index=False),
                       "inventory_recommendations.csv", "text/csv")
    st.caption(f"Showing the top 500 of {len(table):,} records. Predictions: {spec.name}, on the held-out test period.")


# ─────────────────────────────────────────────
# Tab 2: Predict and recommend
# ─────────────────────────────────────────────
def predict_tab(models, spec, rules):
    lk = data.lookups()
    model = predictor.get_model(spec)
    form, result = st.columns([1, 1.15], gap="large")

    with form:
        st.subheader("Scenario")
        a, b = st.columns(2)
        date = a.date_input("Date", dt.date(2026, 12, 31), min_value=lk["date_min"], max_value=dt.date(2027, 12, 31))
        store = b.selectbox("Store", lk["stores"])
        a, b = st.columns(2)
        category = a.selectbox("Category", list(lk["subcategories"]))
        subcategory = b.selectbox("Subcategory", lk["subcategories"][category])
        product = st.selectbox("Product", data.products_for(category, subcategory),
                               help="Used to look up the latest recorded stock for this store.")
        typ = lk["typical"].loc[(category, subcategory)]
        key = f"{store}-{product}-{category}-{subcategory}"       # a new choice resets the defaults below

        latest = data.latest_inventory(store, product, category, subcategory)
        inv_default = latest[0] if latest else int(typ["inventory"])
        a, b = st.columns(2)
        inventory = a.number_input("Stock on hand (units)", 0, 5000, inv_default, key=f"inv-{key}")
        b.caption("")
        b.caption(f"Latest record: {latest[1]}" if latest else "No record for this store/product: using the category median.")

        a, b = st.columns(2)
        price = a.number_input("Unit price", 0.5, 5000.0, float(round(typ["price"], 2)), step=1.0, key=f"price-{category}-{subcategory}")
        discount = b.select_slider("Discount %", lk["discounts"], value=0)
        a, b = st.columns(2)
        promotion = a.toggle("Promotion running", value=discount > 0, key=f"promo-{discount}")
        holiday = b.toggle("Holiday")

        with st.expander("More conditions"):
            a, b = st.columns(2)
            weather = a.selectbox("Weather", lk["weather"])
            store_type = b.selectbox("Store type", lk["store_types"], index=lk["store_types"].index(lk["store_type"][store]), key=f"st-{store}")
            a, b = st.columns(2)
            marketing = a.number_input("Marketing spend", 0.0, 20000.0, float(round(typ["marketing"], 0)), step=100.0, key=f"mk-{category}-{subcategory}")
            footfall = b.number_input("Customer footfall", 0, 3000, int(typ["footfall"]), step=10, key=f"ff-{category}-{subcategory}")
            competitor = st.number_input("Competitor price", 0.5, 5000.0, float(round(price / typ["price_gap"], 2)), step=1.0, key=f"cp-{key}-{price}")

    values = dict(month=date.month, unit_price=price, discount=discount, promotion=promotion, holiday=holiday,
                  marketing=marketing, footfall=footfall, competitor_price=competitor, category=category,
                  subcategory=subcategory, weather=weather, store_type=store_type)
    X = predictor.encode(predictor.build_inputs(**values))
    pred = float(model.predict(X)[0])
    res = rec.assess([pred], [inventory], rules).iloc[0]

    with result:
        st.subheader("Recommendation")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted demand", f"{pred:.1f} units")
        m2.metric("Stock on hand", f"{inventory:,}")
        m3.metric("Days of cover", f"{res['Cover_Days']:.1f}")

        msg = f"**{res['Status']}.** {res['Action']}."
        {"Understocking": st.error, "Overstocking": st.warning, "Sufficient stock": st.success}[res["Status"]](msg, icon=STATUS_ICONS[res["Status"]])
        if res["High_Demand"]:
            st.info(f"⭐ High-demand alert: {pred:.0f} units is above the {rules.high_demand_units:.0f}-unit threshold.")

        st.altair_chart(charts.stock_vs_demand(inventory, pred, rules), **STRETCH)

        st.markdown("**What if the discount changes?**")
        sweep = []
        for d in lk["discounts"]:
            v = {**values, "discount": d}
            sweep.append({"Discount": d, "Predicted": float(model.predict(predictor.encode(predictor.build_inputs(**v)))[0])})
        st.altair_chart(charts.what_if_line(pd.DataFrame(sweep), "Discount", "Discount %"), **STRETCH)

        with st.expander("Compare with the other models"):
            rows = []
            for m in models:
                p = float(predictor.get_model(m).predict(X)[0])
                rows.append({"Model": m.name, "Predicted units": round(p, 1), "Status": rec.label(rec.assess([p], [inventory], rules).iloc[0]["Status"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)


# ─────────────────────────────────────────────
# Tab 3: Model comparison
# ─────────────────────────────────────────────
def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_res, ss_tot = (err ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {"MAE": np.abs(err).mean(), "MSE": (err ** 2).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "R²": 1 - ss_res / ss_tot, "WAPE": np.abs(err).sum() / y.sum(), "Bias": err.mean()}


def grouped_importance(model, X, y, n_repeats=3, sample=2000, seed=42):
    """Increase in RMSE when one feature (all its one-hot columns together) is shuffled."""
    rng = np.random.default_rng(seed)
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    ys = y.loc[Xs.index].to_numpy(float)
    base = np.sqrt(np.mean((model.predict(Xs) - ys) ** 2))
    out = {}
    for f in FEATURES:
        cols = [c for c in Xs.columns if c == f or c.startswith(f + "_")]
        inc = []
        for _ in range(n_repeats):
            Xp = Xs.copy()
            Xp[cols] = Xs[cols].iloc[rng.permutation(len(Xs))].to_numpy()
            inc.append(np.sqrt(np.mean((model.predict(Xp) - ys) ** 2)) - base)
        out[f] = float(np.mean(inc))
    return pd.DataFrame({"Feature": list(out), "RMSE increase": list(out.values())}).sort_values("RMSE increase", ascending=False)


def models_tab(models, meta, X, y, spec):
    rows = {m.name: score(y, predictor.predict_test(m, X)) for m in models}
    table = pd.DataFrame(rows).T.sort_values("RMSE")
    st.subheader("Test-period scores")
    st.caption("Every model is scored on the same held-out latest dates. Lower is better for MAE, MSE, RMSE, WAPE; higher for R².")
    st.dataframe(table.style.format({"MAE": "{:.3f}", "MSE": "{:.2f}", "RMSE": "{:.3f}", "R²": "{:.4f}", "WAPE": "{:.1%}", "Bias": "{:+.3f}"})
                 if hasattr(table, "style") else table, **STRETCH)
    a, b, c = st.columns(3)
    a.altair_chart(charts.metric_bars(table, "RMSE", "RMSE (units)"), **STRETCH)
    b.altair_chart(charts.metric_bars(table, "MAE", "MAE (units)"), **STRETCH)
    c.altair_chart(charts.metric_bars(table, "R²", "R²", lower_better=False), **STRETCH)
    st.success(f"Best model by RMSE: **{table.index[0]}** (R² {table.iloc[0]['R²']:.3f}, MAE {table.iloc[0]['MAE']:.2f} units).")

    st.divider()
    st.subheader(f"{spec.name}: error analysis")
    pred = predictor.predict_test(spec, X)
    a, b = st.columns(2)
    sample = np.random.default_rng(0).choice(len(y), min(3000, len(y)), replace=False)
    a.markdown("**Actual vs predicted**")
    a.altair_chart(charts.actual_vs_predicted(y.to_numpy()[sample], pred[sample]), **STRETCH)
    b.markdown("**Residual distribution**")
    b.altair_chart(charts.residual_hist(y.to_numpy(), pred), **STRETCH)

    a, b = st.columns(2)
    seg = (pd.DataFrame({"Category": meta["Product_Category"], "AbsError": np.abs(pred - y.to_numpy())})
           .groupby("Category")["AbsError"].mean().reset_index().rename(columns={"AbsError": "MAE"}))
    a.markdown("**Error by category (MAE)**")
    a.altair_chart(charts.bar_h(seg, "Category", "MAE", "MAE (units)"), **STRETCH)
    b.markdown("**Feature importance** (RMSE increase when shuffled)")
    imp = grouped_importance(predictor.get_model(spec), X, y)
    b.altair_chart(charts.bar_h(imp, "Feature", "RMSE increase", "RMSE increase (units)", "#2a7ab0"), **STRETCH)


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
    st.session_state["rmse_by_model"] = {m.name: score(y, predictor.predict_test(m, X))["RMSE"] for m in models}
    spec, rules, period = sidebar(models, data.train_sales_distribution(), meta[DATE_COL])

    skipped = [m.name for m in MODELS if m not in models]
    if skipped:
        st.sidebar.warning("Not trained yet: " + ", ".join(skipped))

    tab1, tab2, tab3 = st.tabs(["📊 Inventory overview", "🔮 Predict & recommend", "🧪 Model comparison"])
    with tab1:
        overview_tab(meta, X, spec, rules, period)
    with tab2:
        predict_tab(models, spec, rules)
    with tab3:
        models_tab(models, meta, X, y, spec)