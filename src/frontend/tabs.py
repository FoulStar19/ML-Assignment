"""One function per dashboard tab. Shared values (plan, recommendations, settings) arrive in `ctx`."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from charts import (actions_chart, line_detail, status_by_store, status_strip, stock_vs_reorder)
from config import (ACTION_ORDER, BLUE, CLEAR_EXCESS_SHARE, CURRENCY, INK, MIN_TRANSFER, MUTED,
                    PRIORITY_COLORS, PRIORITY_ORDER, PRIORITY_TINT, STATUS_COLORS, STATUS_ORDER)
from theme import kpi_row, money, section, show, style, styled


def tab_overview(ctx):
    """Headline numbers, status split and the top priorities."""
    asof_date = ctx.asof_date
    excess_value = ctx.excess_value
    lines = ctx.lines
    moved_units = ctx.moved_units
    n_critical = ctx.n_critical
    n_healthy = ctx.n_healthy
    n_lines = ctx.n_lines
    n_over = ctx.n_over
    n_under = ctx.n_under
    net_units = ctx.net_units
    net_value = ctx.net_value
    over_days = ctx.over_days
    recs = ctx.recs
    if n_under == 0:
        headline = f"No lines are at or below their reorder point on {asof_date:%d %b %Y}."
    else:
        crit_txt = f", {n_critical} of them critical" if n_critical else ""
        headline = (f"{n_under} of {n_lines} lines are at or below their reorder point "
                    f"on {asof_date:%d %b %Y}{crit_txt}.")
    st.markdown(f'<div class="lede">{headline}</div>', unsafe_allow_html=True)
    if moved_units:
        sub = (f"Moving surplus between stores covers {moved_units:,} units, leaving a net purchase of "
               f"{net_units:,.0f} units worth about {money(net_value)}.")
    else:
        sub = f"Bringing them back to target level takes {net_units:,.0f} units worth about {money(net_value)}."
    st.markdown(f'<div class="lede-sub">{sub}</div>', unsafe_allow_html=True)

    kpi_row([
        ("Critical lines", f"{n_critical}", "Stock covers less than half the lead time", "bad" if n_critical else ""),
        ("Lines to order", f"{int((recs['net_order'] > 0).sum())}", f"{net_units:,.0f} units to buy", ""),
        ("Net purchase value", money(net_value), "After transfers" if moved_units else "At recent prices", ""),
        ("Excess stock value", money(excess_value), f"{n_over} lines above target level", ""),
        ("Healthy lines", f"{n_healthy / n_lines:.0%}", f"{n_healthy} of {n_lines} lines", "good"),
    ])

    show(status_strip(lines))
    st.markdown(
        '<div class="legend">'
        f'<span><span class="swatch" style="background:{STATUS_COLORS["Understock risk"]}"></span>'
        'Understock risk: stock at or below the reorder point</span>'
        f'<span><span class="swatch" style="background:{STATUS_COLORS["Healthy"]}"></span>Healthy</span>'
        f'<span><span class="swatch" style="background:{STATUS_COLORS["Overstock"]}"></span>'
        f'Overstock: more than {over_days:g} days of cover and above target level</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        show(stock_vs_reorder(lines))
    with right:
        show(status_by_store(lines))

    section("Top priorities", "The six lines to act on first. Open the Recommendations tab for the full list.")
    top = recs[recs["priority"] != "None"].head(6)
    if top.empty:
        st.info("Nothing needs action for the selected stores.")
    else:
        top_view = top[["priority", "action", "store_id", "product_id", "net_order", "transfer"]].rename(columns={
            "priority": "Priority", "action": "Action", "store_id": "Store", "product_id": "Product",
            "net_order": "Order qty", "transfer": "Transfer",
        })
        st.dataframe(
            styled(top_view, {"Order qty": "{:,.0f}"}, tint_col="Priority"),
            hide_index=True, height=35 * (len(top_view) + 1) + 3,
        )


def tab_recommendations(ctx):
    """Ranked actions, the transfer plan and a per-line detail view."""
    asof_date = ctx.asof_date
    moved_units = ctx.moved_units
    moved_value = ctx.moved_value
    plan = ctx.plan
    recs = ctx.recs
    transfers = ctx.transfers
    use_transfers = ctx.use_transfers
    section("Recommendations", "Rule-based and ranked by urgency. Each line gets one action and a reason.")
    a, b = st.columns([2, 3])
    with a:
        if (recs["action"].isin(ACTION_ORDER)).any():
            show(actions_chart(recs))
        else:
            st.info("No actions are needed for the selected stores.")
    with b:
        section("Transfer plan", "Surplus above target at one store covers a shortfall at another, which cuts new orders.")
        if transfers.empty:
            st.info("No transfers are suggested." if use_transfers else "Transfers are switched off in the sidebar.")
        else:
            tp = transfers.sort_values("qty", ascending=False).rename(columns={
                "product_id": "Product", "from_store": "From", "to_store": "To", "qty": "Units", "value": "Value",
            })
            st.dataframe(
                styled(tp, {"Units": "{:,.0f}", "Value": f"{CURRENCY}{{:,.0f}}"}),
                hide_index=True, height=min(35 * (len(tp) + 1) + 3, 353),
            )
            st.caption(
                f"{len(tp)} moves covering {moved_units:,} units, about {money(moved_value)} of purchases avoided. "
                "Transfer cost and shipping time are not modelled."
            )

    f1, f2, f3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    with f1:
        prio_sel = st.multiselect("Priority", PRIORITY_ORDER[:4], default=PRIORITY_ORDER[:4])
    with f2:
        show_none = st.toggle("Include lines with no action", value=False)
    view = recs[recs["priority"].isin(prio_sel + (["None"] if show_none else []))].reset_index(drop=True)
    table = view[[
        "priority", "action", "store_id", "product_id", "inventory_level", "days_of_cover",
        "net_order", "transfer", "why",
    ]].rename(columns={
        "priority": "Priority", "action": "Action", "store_id": "Store", "product_id": "Product",
        "inventory_level": "On hand", "days_of_cover": "Days of cover", "net_order": "Order qty",
        "transfer": "Transfer", "why": "Why",
    })
    with f3:
        st.download_button(
            "Download actions (CSV)", table.round(1).to_csv(index=False),
            file_name=f"actions_{asof_date:%Y-%m-%d}.csv", mime="text/csv",
        )

    if table.empty:
        st.info("No lines match these filters.")
    else:
        st.caption("Select a row to see the full recommendation and its stock history.")
        event = st.dataframe(
            styled(table, {"On hand": "{:,.0f}", "Days of cover": "{:.1f}"}, tint_col="Priority"),
            hide_index=True, key="rec_table", on_select="rerun", selection_mode="single-row",
            height=min(35 * (len(table) + 1) + 3, 423),
            column_config={
                "Priority": st.column_config.Column("Priority", width="small"),
                "Action": st.column_config.Column("Action", width="medium"),
                "Store": st.column_config.Column("Store", width="small"),
                "Product": st.column_config.Column("Product", width="small"),
                "On hand": st.column_config.Column("On hand", width="small"),
                "Days of cover": st.column_config.Column("Cover (days)", width="small"),
                "Order qty": st.column_config.ProgressColumn(
                    "Order qty", format="%d", width="small",
                    min_value=0, max_value=float(max(table["Order qty"].max(), 1)),
                ),
                "Transfer": st.column_config.Column("Transfer", width="medium"),
                "Why": st.column_config.TextColumn("Why", width="large"),
            },
        )
        chosen = event.selection.rows[0] if event.selection.rows else 0
        chosen = chosen if chosen < len(view) else 0
        row = view.loc[chosen]
        colour = PRIORITY_COLORS.get(row["priority"], BLUE)
        tint_bg, tint_fg = PRIORITY_TINT.get(row["priority"], ("#E3E9EF", "#34495E"))
        prio_chip = (f'<span class="rec-prio" style="background:{tint_bg};color:{tint_fg}">{row["priority"]}</span>'
                     if row["priority"] != "None" else "")

        section(f"Line detail: store {row['store_id']}, product {row['product_id']}")
        st.markdown(
            f'<div class="rec-card" style="border-left-color:{colour}">'
            f'<div class="rec-head">{prio_chip}{row["action"]}</div><div>{row["why"]}</div></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("On hand", f"{row['inventory_level']:,.0f}")
        m2.metric("Reorder point", f"{row['reorder_point']:,.0f}")
        m3.metric("Target level", f"{row['target_level']:,.0f}")
        m4.metric("Order qty", f"{row['net_order']:,.0f}")
        show(line_detail(plan, row["store_id"], row["product_id"], asof_date))

    with st.expander("How recommendations are made"):
        st.markdown(
            "Expected daily demand is the average `demand_forecast` over the look-back window. Safety stock is "
            "`z × σ × √lead time`, where σ is the spread of recent forecast errors and z comes from the service level. "
            "The reorder point is `demand × lead time + safety stock`. The target level covers lead time plus the days "
            "between orders, with the same safety margin. Stock on hand is treated as the full inventory position, "
            "so units already on order are not deducted."
        )
        st.markdown(
            "A line at or below its reorder point is **Understock risk**. It is **Critical** when its stock covers less "
            "than half the lead time, otherwise **High**. A line is **Overstock** when stock lasts longer than the "
            "overstock limit and sits above target. Overstock is **Medium** when surplus can be transferred or exceeds "
            f"{CLEAR_EXCESS_SHARE:.0%} of target (clear it), and **Low** when ordering should simply pause. "
            "Healthy lines close to their reorder point get a **Low** “reorder soon” heads-up."
        )
        st.markdown(
            f"Transfers match surplus above a store's target to shortfalls for the same product at other stores, "
            f"largest shortfall first, in moves of at least {MIN_TRANSFER} units. The remaining shortfall is ordered."
        )


def tab_demand(ctx):
    """Forecast accuracy and demand against forecast."""
    hist = ctx.hist
    if hist.empty:
        st.info("No rows match the current filters.")
    else:
        err = hist["forecast_error"]
        mae = err.abs().mean()
        wape = err.abs().sum() / max(hist["units_sold"].sum(), 1)
        bias = err.mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Average miss", f"{mae:.1f} units", help="Mean absolute error per store-product-day.")
        c2.metric("Error as share of sales", f"{wape:.1%}", help="Total absolute error divided by total units sold (WAPE).")
        c3.metric(
            "Average bias", f"{bias:+.1f} units",
            help="Units sold minus forecast. Negative means forecasts ran higher than sales, which pushes stock toward overstock.",
        )

        grain = st.radio("Group by", ["Day", "Week", "Month"], index=1, horizontal=True)
        freq = {"Day": "D", "Week": "W", "Month": "MS"}[grain]
        grouper = pd.Grouper(key="date", freq=freq)
        # Average per day, so partial weeks or months at the edges don't distort the line.
        series = hist.groupby(grouper)[["units_sold", "demand_forecast"]].sum()
        days = hist.groupby(grouper)["date"].nunique().replace(0, np.nan)
        series = series.div(days, axis=0).dropna()
        gap = series["demand_forecast"] - series["units_sold"]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
        fig.add_scatter(x=series.index, y=series["units_sold"], name="Units sold",
                        line=dict(color=INK, width=2), row=1, col=1)
        fig.add_scatter(x=series.index, y=series["demand_forecast"], name="Forecast",
                        line=dict(color=BLUE, width=2, dash="dash"), row=1, col=1)
        fig.add_bar(
            x=gap.index, y=gap, name="Forecast minus sales", showlegend=False,
            marker_color=[STATUS_COLORS["Overstock"] if v >= 0 else STATUS_COLORS["Understock risk"] for v in gap],
            hovertemplate="%{y:+,.0f} units<extra>Forecast minus sales</extra>", row=2, col=1,
        )
        style(fig, height=440, title=f"Units per day, averaged by {grain.lower()}", legend=True)
        fig.update_yaxes(title="Units per day", row=1, col=1)
        fig.update_yaxes(title="Forecast minus sales", row=2, col=1)
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=0.27, xanchor="left", yanchor="bottom", showarrow=False,
            text="Above zero: forecast ran high, stock builds up. Below zero: forecast ran low, stockouts follow.",
            font=dict(size=12, color=MUTED),
        )
        show(fig)

        a, b = st.columns(2)
        with a:
            fig = px.histogram(hist, x="forecast_error", nbins=30, color_discrete_sequence=[BLUE])
            fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
            style(fig, height=320, title="Forecast error (units sold minus forecast)")
            fig.update_xaxes(title="Units")
            fig.update_yaxes(title="Store-product-days")
            show(fig)
        with b:
            acc_chart = st.container()
            dim = st.selectbox(
                "Break accuracy down by", ["category", "store_id", "region"],
                format_func=lambda c: {"category": "Category", "store_id": "Store", "region": "Region"}[c],
            )
            acc = hist.groupby(dim).apply(
                lambda g: g["forecast_error"].abs().sum() / max(g["units_sold"].sum(), 1),
                include_groups=False,
            ).reset_index(name="wape")
            fig = px.bar(acc, x=dim, y="wape", text_auto=".1%", color_discrete_sequence=[BLUE])
            fig.update_traces(textposition="outside", cliponaxis=False)
            style(fig, height=262, title="Error as share of sales")
            fig.update_xaxes(title="")
            fig.update_yaxes(visible=False, showgrid=False)
            with acc_chart:
                show(fig)


def tab_health(ctx):
    """How stock status has moved over time and where it is weakest."""
    hist = ctx.hist
    if hist.empty:
        st.info("No rows match the current filters.")
    else:
        monthly = (
            hist.assign(month=hist["date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "status"]).size().unstack(fill_value=0)
            .reindex(columns=STATUS_ORDER, fill_value=0)
        )
        monthly = monthly.div(monthly.sum(axis=1), axis=0).reset_index().melt(
            id_vars="month", var_name="status", value_name="share"
        )
        fig = px.bar(monthly, x="month", y="share", color="status",
                     color_discrete_map=STATUS_COLORS, category_orders={"status": STATUS_ORDER})
        style(fig, height=340, title="Share of store-product days by stock status", legend=True)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", tickformat=".0%", range=[0, 1])
        show(fig)

        left, right = st.columns(2)
        with left:
            heat_chart = st.container()
            measure = st.selectbox("Heatmap measure", ["Understock rate", "Overstock rate", "Average days of cover"])
            if measure == "Understock rate":
                values, fmt, colour = (hist["status"] == "Understock risk").astype(float), ".0%", STATUS_COLORS["Understock risk"]
            elif measure == "Overstock rate":
                values, fmt, colour = (hist["status"] == "Overstock").astype(float), ".0%", STATUS_COLORS["Overstock"]
            else:
                values, fmt, colour = hist["days_of_cover"], ".1f", BLUE
            pivot = hist.assign(v=values).pivot_table(index="store_id", columns="category", values="v", aggfunc="mean")
            # Anchor the scale at zero so small differences look small.
            fig = px.imshow(
                pivot, text_auto=fmt, aspect="auto",
                color_continuous_scale=["#FFFFFF", colour], zmin=0, zmax=float(pivot.max().max()) * 1.3,
            )
            fig.update_coloraxes(showscale=False)
            style(fig, height=320, title=f"{measure} by store and category")
            fig.update_xaxes(title="", side="bottom")
            fig.update_yaxes(title="")
            with heat_chart:
                show(fig)
        with right:
            section("Lines most often understocked", "Share of days each line sat at or below its reorder point.")
            risk = (
                hist.assign(
                    under=(hist["status"] == "Understock risk") * 100.0,
                    over=(hist["status"] == "Overstock") * 100.0,
                )
                .groupby(["store_id", "product_id"], as_index=False)
                .agg(under=("under", "mean"), over=("over", "mean"), cover=("days_of_cover", "mean"))
                .sort_values("under", ascending=False).head(10)
                .rename(columns={"store_id": "Store", "product_id": "Product", "under": "Understock",
                                 "over": "Overstock", "cover": "Cover (days)"})
            )
            st.dataframe(
                risk, hide_index=True,
                column_config={
                    "Understock": st.column_config.NumberColumn(format="%.0f%%"),
                    "Overstock": st.column_config.NumberColumn(format="%.0f%%"),
                    "Cover (days)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
