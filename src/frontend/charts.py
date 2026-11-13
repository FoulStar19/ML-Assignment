import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (ACTION_ORDER, BLUE, DETAIL_DAYS, INK, MUTED, PRIORITY_COLORS, PRIORITY_ORDER,
                    STATUS_COLORS, STATUS_ORDER, STATUS_TEXT)
from theme import style


def status_strip(lines: pd.DataFrame) -> go.Figure:
    counts = lines["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig = go.Figure()
    for status, n in counts.items():
        fig.add_bar(
            x=[n], y=[""], orientation="h", showlegend=False,
            marker_color=STATUS_COLORS[status],
            text=[f"<b>{n}</b>"] if n else None,
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color=STATUS_TEXT[status], size=15),
            hovertemplate=f"{status}: {n} lines<extra></extra>",
        )
    style(fig, height=58)
    fig.update_layout(barmode="stack", bargap=0, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False, range=[0, max(int(counts.sum()), 1)])
    fig.update_yaxes(visible=False)
    return fig


def stock_vs_reorder(lines: pd.DataFrame) -> go.Figure:
    x_max = float(lines["reorder_point"].max()) * 1.15
    y_max = float(lines["inventory_level"].max()) * 1.05
    diag = min(x_max, y_max)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, diag], y=[0, diag], mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color="#9AA7B4", dash="dot", width=1),
    ))
    for status in STATUS_ORDER:
        s = lines[lines["status"] == status]
        fig.add_trace(go.Scatter(
            x=s["reorder_point"], y=s["inventory_level"], mode="markers", name=status,
            marker=dict(color=STATUS_COLORS[status], size=9, line=dict(color="white", width=1)),
            customdata=s[["store_id", "product_id", "order_qty"]].to_numpy(),
            hovertemplate=(
                "Store %{customdata[0]}, product %{customdata[1]}<br>"
                "On hand %{y:,.0f}, reorder point %{x:,.0f}<br>"
                "Order %{customdata[2]:,.0f}<extra></extra>"
            ),
        ))
    fig.add_annotation(
        x=x_max, y=0, xanchor="right", yanchor="bottom", showarrow=False,
        text="Below the dotted line: order needed", font=dict(size=12, color=MUTED),
    )
    style(fig, title="Stock on hand vs reorder point")
    fig.update_xaxes(title="Reorder point (units)", range=[0, x_max])
    fig.update_yaxes(title="Stock on hand (units)", range=[0, y_max])
    return fig


def status_by_store(lines: pd.DataFrame) -> go.Figure:
    counts = (
        lines.groupby(["store_id", "status"]).size().unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0).reset_index()
        .melt(id_vars="store_id", var_name="status", value_name="lines")
    )
    counts["label"] = counts["lines"].astype(str).where(counts["lines"] > 0, "")
    fig = px.bar(
        counts, y="store_id", x="lines", color="status", orientation="h", text="label",
        color_discrete_map=STATUS_COLORS,
        category_orders={"status": STATUS_ORDER, "store_id": sorted(counts["store_id"].unique())},  # px lists the first item at the top of horizontal bars
    )
    for trace in fig.data:
        trace.textfont.color = STATUS_TEXT[trace.name]
        trace.textposition = "inside"
        trace.insidetextanchor = "middle"
    style(fig, title="Stock status by store")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title="", showgrid=False)
    fig.update_layout(bargap=0.3)
    return fig


def actions_chart(recs: pd.DataFrame) -> go.Figure:
    acted = recs[recs["action"].isin(ACTION_ORDER)]
    counts = acted.groupby(["action", "priority"]).size().reset_index(name="lines")
    fig = px.bar(
        counts, y="action", x="lines", color="priority", orientation="h", text="lines",
        color_discrete_map=PRIORITY_COLORS,
        category_orders={"action": ACTION_ORDER, "priority": PRIORITY_ORDER[:4]},
    )
    for trace in fig.data:
        trace.textfont.color = "white" if trace.name != "Medium" else INK
        trace.textposition = "inside"
        trace.insidetextanchor = "middle"
    style(fig, height=330, title="Lines by recommended action", legend=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title="", showgrid=False)
    fig.update_layout(bargap=0.3)
    return fig


def line_detail(plan: pd.DataFrame, store: str, product: str, asof: pd.Timestamp) -> go.Figure:
    h = plan[(plan["store_id"] == store) & (plan["product_id"] == product) & (plan["date"] <= asof)].tail(DETAIL_DAYS)
    low = h[h["status"] == "Understock risk"]
    red = STATUS_COLORS["Understock risk"]
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["target_level"], name="Target level",
                    line=dict(color=BLUE, width=1.5, dash="dot"))
    fig.add_scatter(x=h["date"], y=h["reorder_point"], name="Reorder point",
                    line=dict(color=red, width=1.5, dash="dash"))
    fig.add_scatter(x=h["date"], y=h["units_sold"], name="Units sold",
                    line=dict(color="#A9B5C1", width=1.2))
    fig.add_scatter(x=h["date"], y=h["inventory_level"], name="Stock on hand",
                    line=dict(color=INK, width=2))
    fig.add_scatter(x=low["date"], y=low["inventory_level"], name="Order day", mode="markers",
                    marker=dict(color=red, size=8, line=dict(color="white", width=1)))
    style(fig, height=330, title=f"Store {store}, product {product}: last {len(h)} days", legend=True)
    fig.update_yaxes(title="Units", rangemode="tozero")
    return fig


