"""One function per dashboard tab. Shared values (plan, recommendations, settings) arrive in `ctx`.

Selecting a line anywhere (map, table or the pickers in the inspector) updates one shared choice,
st.session_state["sel_line"], so every view shows the same line.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from charts import line_detail, stock_map, stock_vs_reorder
from config import (ACTION_ORDER, BLUE, CURRENCY, INK, MUTED, PRIORITY_COLORS, PRIORITY_ORDER,
                    PRIORITY_TINT, STATUS_COLORS, STATUS_ORDER)
from theme import chip, kpi_row, money, prob_bar, section, show, style, styled

STATUS_TINT = {"Understock risk": ("#F7DDE0", "#8E2434"), "Healthy": ("#D9EEE9", "#1B5A4F"), "Overstock": ("#F8E8C8", "#7A5410")}


# ─────────────────────────────────────────────
# Shared line selection
# ─────────────────────────────────────────────
def _selected(ctx) -> tuple[str, str]:
    """The chosen (store, product), or the top priority line if nothing valid is chosen."""
    sel = st.session_state.get("sel_line")
    valid = set(zip(ctx.recs["store_id"], ctx.recs["product_id"]))
    if sel not in valid:
        top = ctx.recs.iloc[0]
        sel = (top["store_id"], top["product_id"])
        st.session_state["sel_line"] = sel
    return sel


def _pick_from_map(key: str) -> None:
    pts = st.session_state.get(key, {}).get("selection", {}).get("points", [])
    if pts:
        st.session_state["sel_line"] = (pts[0]["y"], pts[0]["x"])      # y = store, x = product


def _pick_from_pickers(key: str) -> None:
    st.session_state["sel_line"] = (st.session_state[f"{key}_store"], st.session_state[f"{key}_product"])


def _pick_from_table(key: str, frame: pd.DataFrame) -> None:
    rows = st.session_state.get(key, {}).get("selection", {}).get("rows", [])
    if rows and rows[0] < len(frame):
        r = frame.iloc[rows[0]]
        st.session_state["sel_line"] = (r["Store"], r["Product"])


# ─────────────────────────────────────────────
# Line inspector (used on Overview and Actions)
# ─────────────────────────────────────────────
def inspector(ctx, key: str) -> None:
    """Model verdict, probabilities, recommended action, an order what-if and the stock history."""
    store, product = _selected(ctx)
    stores = sorted(ctx.recs["store_id"].unique())
    products = sorted(ctx.recs["product_id"].unique())
    # Keep both pickers on the shared choice (a map or table click changes it); their own changes come back via on_change.
    st.session_state[f"{key}_store"], st.session_state[f"{key}_product"] = store, product
    c1, c2 = st.columns(2)
    c1.selectbox("Store", stores, key=f"{key}_store", on_change=_pick_from_pickers, args=(key,))
    c2.selectbox("Product", products, key=f"{key}_product", on_change=_pick_from_pickers, args=(key,))

    row = ctx.recs[(ctx.recs["store_id"] == store) & (ctx.recs["product_id"] == product)].iloc[0]
    bg, fg = STATUS_TINT[row["status"]]
    prio = row["priority"]
    pbg, pfg = PRIORITY_TINT.get(prio, ("#E3E9EF", "#34495E"))
    chips = chip(row["status"], bg, fg) + (chip(prio, pbg, pfg) if prio != "None" else "")
    st.markdown(
        f'<div class="insp"><div class="title">{row["action"]}</div>'
        f'<div class="sub">{chips}</div>'
        f'{prob_bar(row["p_under"], row["p_healthy"], row["p_over"])}'
        f'<div class="why">{row["why"]}</div></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("On hand", f"{row['inventory_level']:,.0f}")
    m2.metric("Cover", f"{row['days_of_cover']:.1f} d")
    m3.metric("Order", f"{row['net_order']:,.0f}")

    # What-if: change the order and see the cover it buys against the lead time.
    top = int(max(row["target_level"] * 1.5, row["net_order"] * 1.5, 50))
    qty = st.slider("Try an order quantity", 0, top, int(min(row["net_order"], top)), key=f"{key}_qty_{store}_{product}")
    cover_after = (row["inventory_level"] + qty) / row["mu"]
    delta = cover_after - ctx.lead
    st.metric("Cover after order", f"{cover_after:.1f} days",
              f"{delta:+.1f} vs {ctx.lead}-day lead time", delta_color="normal")

    show(line_detail(ctx.plan, store, product, ctx.asof_date), key=f"{key}_history")


# ─────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────
def tab_overview(ctx):
    lines, recs = ctx.lines, ctx.recs
    kpi_row([
        ("Critical", f"{ctx.n_critical}", "Stock below half the lead time", "bad" if ctx.n_critical else "good"),
        ("Understock risk", f"{ctx.n_under}", f"of {ctx.n_lines} lines", "bad" if ctx.n_under else "good"),
        ("To buy", money(ctx.net_value), f"{ctx.net_units:,.0f} units" + (", after transfers" if ctx.moved_units else ""), ""),
        ("Excess stock", money(ctx.excess_value), f"{ctx.n_over} overstocked lines", "warn"),
        ("Healthy", f"{ctx.n_healthy / ctx.n_lines:.0%}", f"{ctx.n_healthy} lines", "good"),
    ])

    counts = lines["status"].value_counts()
    options = ["All"] + STATUS_ORDER
    focus = st.segmented_control(
        "Show", options, default="All", key="focus", label_visibility="collapsed",
        format_func=lambda o: f"All {len(lines)}" if o == "All" else f"{o} {counts.get(o, 0)}",
    ) or "All"

    left, right = st.columns([3, 2], gap="large")
    with left:
        view = st.segmented_control("Chart", ["Stock map", "Stock vs reorder point"], default="Stock map",
                                    key="ov_chart", label_visibility="collapsed") or "Stock map"
        if view == "Stock map":
            st.plotly_chart(
                stock_map(lines, _selected(ctx), focus), theme=None, key="stock_map",
                config={"displayModeBar": False}, on_select=lambda: _pick_from_map("stock_map"),
                selection_mode="points",
            )
            st.caption("Click a square to inspect it.")
        else:
            show(stock_vs_reorder(lines))

        top = recs[recs["priority"] != "None"]
        if focus != "All":
            top = top[top["status"] == focus]
        top = top.head(8)
        section("Top priorities")
        if top.empty:
            st.info("Nothing to act on here.")
        else:
            table = top[["priority", "action", "store_id", "product_id", "net_order"]].rename(columns={
                "priority": "Priority", "action": "Action", "store_id": "Store",
                "product_id": "Product", "net_order": "Order",
            }).reset_index(drop=True)
            st.dataframe(
                styled(table, {"Order": "{:,.0f}"}, tint_col="Priority"), hide_index=True, key="top_tbl",
                on_select=lambda: _pick_from_table("top_tbl", table), selection_mode="single-row",
                height=35 * (len(table) + 1) + 3,
            )
    with right:
        inspector(ctx, "ov")


# ─────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────
def tab_recommendations(ctx):
    recs, transfers = ctx.recs, ctx.transfers
    acted = recs[recs["action"].isin(ACTION_ORDER)]
    prio_counts = acted["priority"].value_counts()
    act_counts = acted["action"].value_counts()

    f1, f2 = st.columns(2)
    prios = f1.pills("Priority", PRIORITY_ORDER[:4], selection_mode="multi", default=PRIORITY_ORDER[:4],
                     format_func=lambda p: f"{p} {prio_counts.get(p, 0)}")
    acts = f2.pills("Action", [a for a in ACTION_ORDER if a in act_counts.index], selection_mode="multi",
                    format_func=lambda a: f"{a} {act_counts.get(a, 0)}")

    view = recs[recs["priority"].isin(prios or [])]
    if acts:
        view = view[view["action"].isin(acts)]
    view = view.reset_index(drop=True)
    table = view[["priority", "action", "store_id", "product_id", "confidence", "inventory_level",
                  "days_of_cover", "net_order", "transfer", "why"]].rename(columns={
        "priority": "Priority", "action": "Action", "store_id": "Store", "product_id": "Product",
        "confidence": "Model", "inventory_level": "On hand", "days_of_cover": "Cover",
        "net_order": "Order", "transfer": "Transfer", "why": "Why",
    })

    left, right = st.columns([3, 2], gap="large")
    with left:
        h1, h2 = st.columns([3, 1], vertical_alignment="center")
        h1.markdown(f"**{len(table)} lines**")
        h2.download_button("Download CSV", table.round(2).to_csv(index=False),
                           file_name=f"actions_{ctx.asof_date:%Y-%m-%d}.csv", mime="text/csv")
        if table.empty:
            st.info("No lines match these filters.")
        else:
            st.dataframe(
                styled(table, {"Model": "{:.0%}", "On hand": "{:,.0f}", "Cover": "{:.1f}"}, tint_col="Priority"),
                hide_index=True, key="rec_table", selection_mode="single-row",
                on_select=lambda: _pick_from_table("rec_table", table),
                height=min(35 * (len(table) + 1) + 3, 560),
                column_config={
                    "Priority": st.column_config.Column(width="small"),
                    "Action": st.column_config.Column(width="medium"),
                    "Store": st.column_config.Column(width="small"),
                    "Product": st.column_config.Column(width="small"),
                    "Model": st.column_config.ProgressColumn("Model", format="percent", min_value=0, max_value=1, width="small"),
                    "Order": st.column_config.ProgressColumn(
                        "Order", format="%d", width="small", min_value=0, max_value=float(max(table["Order"].max(), 1))),
                    "Why": st.column_config.TextColumn(width="large"),
                },
            )
        with st.expander(f"Store transfers ({len(transfers)})"):
            if transfers.empty:
                st.caption("No transfers suggested." if ctx.use_transfers else "Transfers are off in the sidebar.")
            else:
                tp = transfers.sort_values("qty", ascending=False).rename(columns={
                    "product_id": "Product", "from_store": "From", "to_store": "To", "qty": "Units", "value": "Value"})
                st.dataframe(styled(tp, {"Units": "{:,.0f}", "Value": f"{CURRENCY}{{:,.0f}}"}),
                             hide_index=True, height=min(35 * (len(tp) + 1) + 3, 300))
                st.caption(f"{ctx.moved_units:,} units moved, about {money(ctx.moved_value)} less to buy.")
        with st.expander("How it works"):
            st.markdown(
                "The model flags understock or overstock. Order sizes come from the reorder-point rule: "
                "top up to the target level. Transfers move surplus to stores that are short."
            )
    with right:
        inspector(ctx, "ac")


# ─────────────────────────────────────────────
# Trends (demand + stock health)
# ─────────────────────────────────────────────
def tab_trends(ctx):
    hist = ctx.hist
    if hist.empty:
        st.info("No rows match the current filters.")
        return
    mode = st.segmented_control("View", ["Demand", "Stock health"], default="Demand",
                                key="trend_mode", label_visibility="collapsed") or "Demand"
    if mode == "Demand":
        _demand(hist)
    else:
        _health(hist)


def _demand(hist: pd.DataFrame) -> None:
    err = hist["forecast_error"]
    wape = err.abs().sum() / max(hist["units_sold"].sum(), 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Average miss", f"{err.abs().mean():.1f} units")
    c2.metric("Error vs sales", f"{wape:.1%}")
    c3.metric("Bias", f"{err.mean():+.1f} units", help="Units sold minus forecast.")

    grain = st.segmented_control("Group by", ["Day", "Week", "Month"], default="Week", key="grain") or "Week"
    grouper = pd.Grouper(key="date", freq={"Day": "D", "Week": "W", "Month": "MS"}[grain])
    series = hist.groupby(grouper)[["units_sold", "demand_forecast"]].sum()
    days = hist.groupby(grouper)["date"].nunique().replace(0, np.nan)
    series = series.div(days, axis=0).dropna()
    gap = series["demand_forecast"] - series["units_sold"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
    fig.add_scatter(x=series.index, y=series["units_sold"], name="Sold", line=dict(color=INK, width=2), row=1, col=1)
    fig.add_scatter(x=series.index, y=series["demand_forecast"], name="Forecast",
                    line=dict(color=BLUE, width=2, dash="dash"), row=1, col=1)
    fig.add_bar(x=gap.index, y=gap, showlegend=False,
                marker_color=[STATUS_COLORS["Overstock"] if v >= 0 else STATUS_COLORS["Understock risk"] for v in gap],
                hovertemplate="%{y:+,.0f} units<extra>Forecast minus sold</extra>", row=2, col=1)
    style(fig, height=440, legend=True)
    fig.update_yaxes(title="Units per day", row=1, col=1)
    fig.update_yaxes(title="Gap", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    show(fig)

    a, b = st.columns(2)
    with a:
        fig = px.histogram(hist, x="forecast_error", nbins=30, color_discrete_sequence=[BLUE])
        fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
        style(fig, height=300, title="Forecast error")
        fig.update_xaxes(title="Sold minus forecast")
        fig.update_yaxes(title="")
        show(fig)
    with b:
        dim = st.segmented_control("Accuracy by", ["category", "store_id", "region"], default="category",
                                   key="acc_dim", format_func=lambda c: {"category": "Category", "store_id": "Store",
                                                                         "region": "Region"}[c]) or "category"
        acc = hist.groupby(dim).apply(
            lambda g: g["forecast_error"].abs().sum() / max(g["units_sold"].sum(), 1), include_groups=False,
        ).reset_index(name="wape")
        fig = px.bar(acc, x=dim, y="wape", text_auto=".1%", color_discrete_sequence=[BLUE])
        fig.update_traces(textposition="outside", cliponaxis=False)
        style(fig, height=250)
        fig.update_xaxes(title="")
        fig.update_yaxes(visible=False, showgrid=False)
        show(fig)


def _health(hist: pd.DataFrame) -> None:
    """What actually happened: the stock status each day turned out to have over the following 7 days."""
    h = hist.dropna(subset=["stock_status"]).copy()
    h["actual"] = h["stock_status"].replace({"Understock": "Understock risk"})
    monthly = (
        h.assign(month=h["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "actual"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    )
    monthly = monthly.div(monthly.sum(axis=1), axis=0).reset_index().melt(id_vars="month", var_name="status", value_name="share")
    fig = px.bar(monthly, x="month", y="share", color="status", color_discrete_map=STATUS_COLORS,
                 category_orders={"status": STATUS_ORDER})
    style(fig, height=320, title="Actual stock status by month", legend=True)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="", tickformat=".0%", range=[0, 1])
    show(fig)

    left, right = st.columns(2)
    with left:
        measure = st.segmented_control("Heatmap", ["Understock", "Overstock", "Cover days"], default="Understock",
                                       key="heat") or "Understock"
        if measure == "Cover days":
            values, fmt, colour = h["forward_cover_days"].clip(upper=10), ".1f", BLUE
        elif measure == "Understock":
            values, fmt, colour = (h["actual"] == "Understock risk").astype(float), ".0%", STATUS_COLORS["Understock risk"]
        else:
            values, fmt, colour = (h["actual"] == "Overstock").astype(float), ".0%", STATUS_COLORS["Overstock"]
        pivot = h.assign(v=values).pivot_table(index="store_id", columns="category", values="v", aggfunc="mean")
        fig = px.imshow(pivot, text_auto=fmt, aspect="auto", color_continuous_scale=["#FFFFFF", colour],
                        zmin=0, zmax=float(pivot.max().max()) * 1.3)
        fig.update_coloraxes(showscale=False)
        style(fig, height=300)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        show(fig)
    with right:
        section("Most often understocked")
        risk = (
            h.assign(under=(h["actual"] == "Understock risk") * 100.0, over=(h["actual"] == "Overstock") * 100.0)
            .groupby(["store_id", "product_id"], as_index=False)
            .agg(under=("under", "mean"), over=("over", "mean"))
            .sort_values("under", ascending=False).head(8)
            .rename(columns={"store_id": "Store", "product_id": "Product", "under": "Understock", "over": "Overstock"})
        )
        st.dataframe(risk, hide_index=True, column_config={
            "Understock": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Overstock": st.column_config.NumberColumn(format="%.0f%%"),
        })
