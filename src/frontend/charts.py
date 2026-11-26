import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import BLUE, INK, STATUS_COLORS, STATUS_ORDER
from theme import style


def _fade(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def map_cells(lines: pd.DataFrame) -> pd.DataFrame:
    """Place every line in a grid: grouped by status, most confident calls first. Adds columns x (column) and y (row)."""
    order = {s: i for i, s in enumerate(STATUS_ORDER)}
    cells = (lines.assign(_s=lines["status"].map(order))
             .sort_values(["_s", "confidence", "store_id", "product_id"], ascending=[True, False, True, True])
             .drop(columns="_s").reset_index(drop=True))
    ncols = max(8, int(np.ceil(np.sqrt(len(cells) * 2.4))))
    cells["x"] = cells.index % ncols
    cells["y"] = cells.index // ncols
    return cells


def stock_map(lines: pd.DataFrame, selected: tuple | None, focus: str = "All") -> go.Figure:
    """One square per store-product line, coloured by the model's call. Click a square to inspect it.

    Stores and products are too many to use as axes, so the squares fill a grid, grouped by status.
    """
    cells = map_cells(lines)
    nrows, ncols = int(cells["y"].max()) + 1, int(cells["x"].max()) + 1
    size = int(max(6, min(27, 280 / nrows, 500 / ncols) - 2))
    fig = go.Figure()
    for status in STATUS_ORDER:
        s = cells[cells["status"] == status]
        if s.empty:
            continue
        colour = STATUS_COLORS[status] if focus in ("All", status) else _fade(STATUS_COLORS[status], 0.16)
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="markers", name=status,
            marker=dict(symbol="square", size=size, color=colour, line=dict(color="white", width=1)),
            customdata=s[["store_id", "product_id", "confidence", "inventory_level", "order_qty"]].to_numpy(),
            hovertemplate=("<b>%{customdata[0]} · %{customdata[1]}</b><br>" + status + ", %{customdata[2]:.0%} sure<br>"
                           "On hand %{customdata[3]:,.0f}, order %{customdata[4]:,.0f}<extra></extra>"),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
        ))
    if selected is not None:
        sel = cells[(cells["store_id"] == selected[0]) & (cells["product_id"] == selected[1])]
        if not sel.empty:
            fig.add_trace(go.Scatter(
                x=sel["x"], y=sel["y"], mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(symbol="square-open", size=size + 10, color=INK, line=dict(color=INK, width=3)),
                selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
            ))
    style(fig, height=360, legend=True)
    fig.update_layout(margin=dict(l=16, r=16, t=52, b=14), clickmode="event+select")
    fig.update_xaxes(range=[-0.7, ncols - 0.3], visible=False, showgrid=False, fixedrange=True)
    fig.update_yaxes(range=[nrows - 0.3, -0.7], visible=False, showgrid=False, fixedrange=True)
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
                "%{customdata[0]} · %{customdata[1]}<br>"
                "On hand %{y:,.0f} · reorder %{x:,.0f}<extra></extra>"
            ),
        ))
    style(fig, title="Stock vs reorder point")
    fig.update_xaxes(title="Reorder point (units)", range=[0, x_max])
    fig.update_yaxes(title="Stock on hand (units)", range=[0, y_max])
    return fig


def line_detail(row: pd.Series, over_days: float) -> go.Figure:
    """The selected line's stock against its demand and the levels that decide its status."""
    red, amber = STATUS_COLORS["Understock risk"], STATUS_COLORS["Overstock"]
    bars = [
        ("Units sold", row["units_sold"], "#A9B5C1"),
        ("Stock on hand", row["inventory_level"], INK),
        ("Reorder point", row["reorder_point"], red),
        ("Target level", row["target_level"], BLUE),
        ("Overstock level", row["mu"] * over_days, amber),
    ]
    fig = go.Figure(go.Bar(
        x=[b[0] for b in bars], y=[b[1] for b in bars], marker_color=[b[2] for b in bars],
        text=[f"{b[1]:,.0f}" for b in bars], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:,.0f} units<extra></extra>",
    ))
    style(fig, height=330, title=f"{row['store_id']} · {row['product_id']} · {row['date']:%d %b %Y}")
    fig.update_yaxes(title="Units", rangemode="tozero")
    fig.update_xaxes(title="")
    return fig


def status_breakdown(lines: pd.DataFrame, dim: str) -> go.Figure:
    """Lines per status within each group (category, region or store type), the most at-risk group first."""
    counts = lines.groupby([dim, "status"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    counts = counts.assign(_u=counts["Understock risk"] / counts.sum(axis=1)).sort_values("_u").drop(columns="_u")
    fig = go.Figure()
    for status in STATUS_ORDER:
        fig.add_trace(go.Bar(
            y=counts.index, x=counts[status], name=status, orientation="h", marker_color=STATUS_COLORS[status],
            text=counts[status].where(counts[status] > 0, None), textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white" if status != "Overstock" else INK, size=12),
            hovertemplate="%{y}: %{x} lines<extra>" + status + "</extra>",
        ))
    style(fig, height=max(300, 56 * len(counts) + 90), title="Lines by status", legend=True)
    fig.update_layout(barmode="stack", bargap=0.35, legend=dict(traceorder="normal"))
    fig.update_xaxes(title="Store-product lines", showgrid=True, gridcolor="#E4E9F3")
    fig.update_yaxes(title="", showgrid=False)
    return fig
