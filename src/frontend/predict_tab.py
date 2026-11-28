"""Tab 2, Predict & explore: set a scenario on the left; the charts on the right update with it."""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from . import charts, data, predictor, recommender as rec
from .config import STATUS_ICONS
from .ui import STRETCH

Z90 = 1.645                                   # 90% prediction interval


def inputs(lk):
    st.markdown("##### Inputs")
    a, b = st.columns(2)
    date = a.date_input("Date", dt.date(2026, 12, 31), min_value=lk["date_min"], max_value=dt.date(2027, 12, 31))
    store = b.selectbox("Store", lk["stores"])
    a, b = st.columns(2)
    category = a.selectbox("Category", list(lk["subcategories"]))
    subcategory = b.selectbox("Subcategory", lk["subcategories"][category])
    product = st.selectbox("Product", data.products_for(category, subcategory), help="Used to look up the latest recorded stock for this store.")
    typ = lk["typical"].loc[(category, subcategory)]
    key = f"{store}-{product}-{category}-{subcategory}"        # a new choice resets the defaults below

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
    values = dict(month=date.month, unit_price=price, discount=discount, promotion=promotion, holiday=holiday, marketing=marketing,
                  footfall=footfall, competitor_price=competitor, category=category, subcategory=subcategory, weather=weather, store_type=store_type)
    return values, inventory


def sweep_values(variable, values, lk):
    """The values tried on the what-if chart, as (form key, x title, list of values)."""
    if variable == "Unit price":
        return "unit_price", "Unit price", list(np.round(values["unit_price"] * np.linspace(0.6, 1.4, 15), 2))
    if variable == "Discount %":
        return "discount", "Discount %", list(lk["discounts"])
    if variable == "Marketing spend":
        top = max(values["marketing"] * 2, 1000.0)
        return "marketing", "Marketing spend", list(np.round(np.linspace(0, top, 15), 0))
    base = max(values["footfall"], 10)
    return "footfall", "Customer footfall", list(np.round(base * np.linspace(0.5, 1.5, 15)).astype(int))


def render(models, spec, rules, sigma, sigma_all, lead):
    lk = data.lookups()
    model = predictor.get_model(spec)
    form, result = st.columns([1, 2.2], gap="large")
    with form:
        with st.container(border=True):
            values, inventory = inputs(lk)

    # ---- everything below is recomputed whenever an input changes
    X = predictor.encode(predictor.build_inputs(**values))
    pred = float(model.predict(X)[0])
    sd = sigma.get(values["category"], sigma_all)
    lo, hi = max(pred - Z90 * sd, 0.0), pred + Z90 * sd
    combos = [("Neither", False, False), ("Promotion", True, False), ("Holiday", False, True), ("Promotion + Holiday", True, True)]
    p_combo = predictor.predict_values(model, [{**values, "promotion": p, "holiday": h} for _, p, h in combos])
    now = {(p, h): v for (_, p, h), v in zip(combos, p_combo)}
    promo_effect = now[(True, values["holiday"])] / max(now[(False, values["holiday"])], 1e-9) - 1
    others = {m.name: float(predictor.get_model(m).predict(X)[0]) for m in models}
    agree = sum(lo <= v <= hi for v in others.values())
    res = rec.plan([pred], [inventory], lead(values["category"]), sd, rules).iloc[0]

    with result:
        a, b, c = st.columns(3)
        a.metric("Predicted units (per day)", f"{pred:.1f}", f"90% range {lo:.0f} to {hi:.0f}", delta_color="off")
        b.metric("Promotion effect", f"{promo_effect:+.0%}", "promotion on vs off, other inputs fixed", delta_color="off")
        c.metric("Model agreement", f"{agree} of {len(others)}", "models inside the 90% range", delta_color="off")
        msg = (f"**{res['Status']}.** {res['Action']}. Stock {inventory:,}, reorder point {res['Reorder_Point']:,.0f}, "
               f"stockout risk {res['Stockout_Risk']:.0%} (lead time {res['Lead_Days']:.0f} days).")
        {"Understocking": st.error, "Overstocking": st.warning, "Sufficient stock": st.success}[res["Status"]](msg, icon=STATUS_ICONS[res["Status"]])

        left, right = st.columns(2)
        with left, st.container(border=True):
            st.markdown("**Promotion x Holiday scenarios**")
            scen = pd.DataFrame({"Scenario": [n for n, _, _ in combos], "Predicted": p_combo,
                                 "Current": [(p == values["promotion"] and h == values["holiday"]) for _, p, h in combos]})
            st.altair_chart(charts.scenario_bars(scen), **STRETCH)
            st.caption("Blue = the scenario that matches your inputs.")
        with right, st.container(border=True):
            variable = st.selectbox("What-if: change", ["Unit price", "Discount %", "Marketing spend", "Customer footfall"])
            key, title, xs = sweep_values(variable, values, lk)
            ys = predictor.predict_values(model, [{**values, key: x} for x in xs])
            st.altair_chart(charts.what_if_line(pd.DataFrame({title: xs, "Predicted": ys}), title, title), **STRETCH)

        left, right = st.columns(2)
        with left, st.container(border=True):
            st.markdown("**Price x discount heatmap**")
            prices = list(np.round(values["unit_price"] * np.linspace(0.7, 1.3, 7), 2))
            grid = [(p, d) for p in prices for d in lk["discounts"]]
            hm = predictor.predict_values(model, [{**values, "unit_price": p, "discount": d} for p, d in grid])
            st.altair_chart(charts.heatmap(pd.DataFrame({"Discount": [d for _, d in grid], "Price": [f"{p:g}" for p, _ in grid], "Predicted": hm}),
                                           "Discount", "Price", "Discount %", "Unit price"), **STRETCH)
        with right, st.container(border=True):
            st.markdown("**All models side by side**")
            md = pd.DataFrame({"Model": list(others), "Predicted": list(others.values()), "Selected": [n == spec.name for n in others]})
            st.altair_chart(charts.model_bars(md), **STRETCH)
            st.caption("Green = the model chosen in the sidebar.")
