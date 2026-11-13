import pandas as pd
import plotly.graph_objects as go

from config import BLUE, DETAIL_DAYS, INK, MUTED, STATUS_COLORS, STATUS_ORDER
from theme import style


def _fade(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def stock_map(lines: pd.DataFrame, selected: tuple | None, focus: str = "All") -> go.Figure:
    """One square per store-product line, coloured by the model's call. Click a square to inspect it."""
    stores = sorted(lines["store_id"].unique())
    products = sorted(lines["product_id"].unique())
    fig = go.Figure()
    for status in STATUS_ORDER:
        s = lines[lines["status"] == status]
        if s.empty:
            continue
        colour = STATUS_COLORS[status] if focus in ("All", status) else _fade(STATUS_COLORS[status], 0.16)
        fig.add_trace(go.Scatter(
            x=s["product_id"], y=s["store_id"], mode="markers", name=status,
            marker=dict(symbol="square", size=27, color=colour, line=dict(color="white", width=1)),
            customdata=s[["confidence", "inventory_level", "order_qty"]].to_numpy(),
            hovertemplate=("<b>%{y} · %{x}</b><br>" + status + ", %{customdata[0]:.0%} sure<br>"
                           "On hand %{customdata[1]:,.0f}, order %{customdata[2]:,.0f}<extra></extra>"),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
        ))
    if selected is not None:
        sel = lines[(lines["store_id"] == selected[0]) & (lines["product_id"] == selected[1])]
        if not sel.empty:
            fig.add_trace(go.Scatter(
                x=sel["product_id"], y=sel["store_id"], mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(symbol="square-open", size=37, color=INK, line=dict(color=INK, width=3)),
                selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
            ))
    style(fig, height=360, legend=True)
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), clickmode="event+select")
    fig.update_xaxes(categoryorder="array", categoryarray=products, tickvals=products,
                     ticktext=[str(int(p[1:])) if p[1:].isdigit() else p for p in products],
                     title="Product", showgrid=False, fixedrange=True)
    fig.update_yaxes(categoryorder="array", categoryarray=stores, autorange="reversed",
                     title="", showgrid=False, fixedrange=True)
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


def line_detail(plan: pd.DataFrame, store: str, product: str, asof: pd.Timestamp) -> go.Figure:
    h = plan[(plan["store_id"] == store) & (plan["product_id"] == product) & (plan["date"] <= asof)].tail(DETAIL_DAYS)
    low = h[h["inventory_level"] <= h["reorder_point"]]
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
    fig.add_scatter(x=low["date"], y=low["inventory_level"], name="Below reorder point", mode="markers",
                    marker=dict(color=red, size=8, line=dict(color="white", width=1)))
    style(fig, height=330, title=f"Store {store}, product {product}: last {len(h)} days", legend=True)
    fig.update_yaxes(title="Units", rangemode="tozero")
    return fig


