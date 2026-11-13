"""Smart Inventory Replenishment System - Streamlit dashboard.

Reads the processed retail inventory data and produces, for every store-product
line, a stock status, a prioritised recommendation and (where possible) a
store-to-store transfer, driven by a reorder point / order-up-to policy.

Run:   streamlit run app.py
Theme: copy config.toml to .streamlit/config.toml (same folder level as app.py).
"""
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "retail_store_inventory_processed.csv"
RAW_FILE = BASE_DIR / "data" / "raw" / "retail_store_inventory.csv"

CURRENCY = ""  # the dataset has no currency; set e.g. "RM " to prefix every value shown

REQUIRED_COLUMNS = [
    "date", "store_id", "product_id", "category", "region",
    "inventory_level", "units_sold", "demand_forecast", "price",
]

INK = "#1C2B3A"
MUTED = "#5B6B7B"
BLUE = "#2B4C7E"
GRID = "#E1E7ED"
STATUS_COLORS = {"Understock risk": "#C4384A", "Healthy": "#26796B", "Overstock": "#D79A1F"}
STATUS_TEXT = {"Understock risk": "white", "Healthy": "white", "Overstock": INK}
STATUS_ORDER = list(STATUS_COLORS)

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low", "None"]
PRIORITY_COLORS = {"Critical": "#8E1F30", "High": "#C4384A", "Medium": "#D79A1F", "Low": "#8FA0B3"}
# Soft tints for table cells: (background, text)
PRIORITY_TINT = {
    "Critical": ("#F2C4CA", "#6F1423"),
    "High": ("#F7DDE0", "#8E2434"),
    "Medium": ("#F8E8C8", "#7A5410"),
    "Low": ("#E3E9EF", "#34495E"),
}
ACTION_ORDER = [
    "Order now", "Transfer in + order", "Transfer in", "Transfer out",
    "Clear excess", "Pause ordering", "Reorder soon",
]

MIN_HISTORY_DAYS = 7      # days of history a line needs before it gets a plan
MIN_TRANSFER = 10         # smallest store-to-store transfer worth suggesting (units)
CLEAR_EXCESS_SHARE = 0.6  # surplus this far above target level triggers "Clear excess"
DETAIL_DAYS = 90

st.set_page_config(page_title="Smart Inventory Replenishment", page_icon="📦", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{
        font-family: 'IBM Plex Sans', system-ui, sans-serif;
    }}
    h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        letter-spacing: -0.01em;
    }}
    .block-container {{ padding-top: 1.8rem; max-width: 1360px; }}
    footer {{ visibility: hidden; }}

    /* Page header */
    .page-head {{
        display: flex; justify-content: space-between; align-items: flex-end;
        flex-wrap: wrap; gap: 0.75rem 2rem;
        padding-bottom: 0.9rem; margin-bottom: 0.2rem; border-bottom: 1px solid {GRID};
    }}
    .page-head h1 {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 2rem; font-weight: 700; line-height: 1.1; margin: 0; padding: 0;
    }}
    .page-head .sub {{ color: {MUTED}; margin-top: 0.4rem; max-width: 46rem; }}
    .page-head .asof {{ text-align: right; }}
    .page-head .asof .label {{ color: {MUTED}; font-size: 0.85rem; }}
    .page-head .asof .date {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 1.3rem; font-weight: 600; font-variant-numeric: tabular-nums;
    }}

    /* Tabs */
    button[data-baseweb="tab"] p {{ font-size: 1.02rem; font-weight: 500; }}
    [data-baseweb="tab-list"] {{ gap: 1.6rem; }}

    /* Headline, sections, legend */
    .lede {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 1.5rem; font-weight: 600; line-height: 1.25;
        max-width: 48rem; margin: 0.6rem 0 0.4rem 0;
    }}
    .lede-sub {{ max-width: 46rem; margin-bottom: 0.4rem; color: {MUTED}; }}
    .section-title {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 1.05rem; font-weight: 600; margin: 0.8rem 0 0.2rem 0;
    }}
    .section-note {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 0.4rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1.4rem; margin: 0.5rem 0 0.2rem 0;
               color: {MUTED}; font-size: 0.9rem; }}
    .legend .swatch {{ display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px;
                       margin-right: 0.4rem; vertical-align: -0.05rem; }}

    /* KPI row */
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
             border-top: 1px solid {GRID}; border-bottom: 1px solid {GRID}; margin: 0.8rem 0 1rem 0; }}
    .kpi {{ padding: 0.85rem 1.1rem 0.85rem 0; }}
    .kpi + .kpi {{ padding-left: 1.1rem; border-left: 1px solid {GRID}; }}
    .k-label {{ color: {MUTED}; font-size: 0.85rem; }}
    .k-value {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 1.9rem; font-weight: 700; line-height: 1.15; font-variant-numeric: tabular-nums;
    }}
    .k-value.bad {{ color: {PRIORITY_COLORS["High"]}; }}
    .k-value.good {{ color: {STATUS_COLORS["Healthy"]}; }}
    .k-note {{ color: {MUTED}; font-size: 0.85rem; }}

    /* Recommendation card */
    .rec-card {{ background: #FFFFFF; border: 1px solid {GRID}; border-left: 4px solid {BLUE};
                 border-radius: 3px; padding: 0.8rem 1rem; margin: 0.4rem 0 0.8rem 0; }}
    .rec-card .rec-head {{ font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
                           font-weight: 600; font-size: 1.05rem; margin-bottom: 0.2rem; }}
    .rec-card .rec-prio {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; font-size: 0.8rem;
                           font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 3px; margin-right: 0.6rem; }}

    /* Metrics, sidebar */
    [data-testid="stMetricValue"] {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 1.7rem; font-variant-numeric: tabular-nums;
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
    .brand {{ padding: 0.2rem 0 0.6rem 0; }}
    .brand .name {{ font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif;
                    font-weight: 700; font-size: 1.15rem; }}
    .brand .tag {{ color: {MUTED}; font-size: 0.85rem; }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details {{ border-color: {GRID}; }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif; font-weight: 600;
    }}

    @media (max-width: 640px) {{
        .block-container {{ padding-top: 3.6rem; }}
        .page-head h1 {{ font-size: 1.6rem; }}
        .page-head .asof {{ text-align: left; }}
        .lede {{ font-size: 1.3rem; }}
        .kpis {{ grid-template-columns: 1fr 1fr; }}
        .kpi, .kpi + .kpi {{ padding-left: 0; border-left: none; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"


# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────
def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Minimal version of the pipeline, used only if the processed CSV is missing."""
    df = raw.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_")
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["forecast_error"] = df["units_sold"] - df["demand_forecast"]
    return df


@st.cache_data(show_spinner="Loading inventory data...")
def load_data() -> pd.DataFrame:
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])
    elif RAW_FILE.exists():
        df = prepare_raw(pd.read_csv(RAW_FILE))
    else:
        return pd.DataFrame()
    if "forecast_error" not in df.columns and {"units_sold", "demand_forecast"} <= set(df.columns):
        df["forecast_error"] = df["units_sold"] - df["demand_forecast"]
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner="Estimating demand for each store-product line...")
def demand_stats(lookback: int) -> pd.DataFrame:
    """Rolling expected demand (mu) and forecast-error spread (sigma) per line.

    mu uses demand_forecast. To use a trained model instead, replace that
    column's values with the model's predictions and everything below follows.
    sigma uses yesterday's errors and earlier, so a day's plan never sees its own outcome.
    """
    df = load_data()
    g = df.groupby(["store_id", "product_id"], sort=False)
    out = df.copy()
    out["mu"] = g["demand_forecast"].transform(
        lambda s: s.rolling(lookback, min_periods=MIN_HISTORY_DAYS).mean()
    ).clip(lower=1)
    out["sigma"] = g["forecast_error"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=MIN_HISTORY_DAYS).std()
    )
    out["avg_price"] = g["price"].transform(lambda s: s.rolling(lookback, min_periods=1).mean())
    return out.dropna(subset=["mu", "sigma"]).reset_index(drop=True)


def apply_policy(d: pd.DataFrame, lead: int, review: int, service: float, over_days: float) -> pd.DataFrame:
    """Reorder point / order-up-to policy applied to every store-product-day."""
    z = NormalDist().inv_cdf(service)
    d = d.copy()
    d["safety_stock"] = z * d["sigma"] * np.sqrt(lead)
    d["reorder_point"] = d["mu"] * lead + d["safety_stock"]
    d["target_level"] = d["mu"] * (lead + review) + z * d["sigma"] * np.sqrt(lead + review)
    d["days_of_cover"] = d["inventory_level"] / d["mu"]

    under = d["inventory_level"] <= d["reorder_point"]
    over = (d["days_of_cover"] > over_days) & (d["inventory_level"] > d["target_level"])
    d["status"] = np.select([under, over], ["Understock risk", "Overstock"], default="Healthy")

    d["order_qty"] = np.where(under, np.ceil(d["target_level"] - d["inventory_level"]).clip(lower=0), 0)
    d["order_value"] = d["order_qty"] * d["avg_price"]
    d["excess_units"] = np.where(over, np.floor(d["inventory_level"] - d["target_level"]).clip(lower=0), 0)
    d["excess_value"] = d["excess_units"] * d["avg_price"]
    return d


# ─────────────────────────────────────────────
# 2. RECOMMENDER
# ─────────────────────────────────────────────
def plan_transfers(lines: pd.DataFrame) -> pd.DataFrame:
    """Match surplus stock to shortfalls for the same product across stores.

    Surplus = units above a store's target level. Shortfall = units needed to
    get a store back to target. Largest shortfalls are served first.
    """
    cols = ["product_id", "from_store", "to_store", "qty", "value"]
    rows = []
    for product, g in lines.groupby("product_id"):
        need = g[g["order_qty"] > 0].sort_values("order_qty", ascending=False)
        surplus = g[g["excess_units"] > 0]
        if need.empty or surplus.empty:
            continue
        avail = dict(zip(surplus["store_id"], surplus["excess_units"].astype(int)))
        for _, n in need.iterrows():
            remaining = int(n["order_qty"])
            for src in sorted(avail, key=avail.get, reverse=True):
                qty = min(remaining, avail[src])
                if qty < MIN_TRANSFER:
                    continue
                rows.append({"product_id": product, "from_store": src, "to_store": n["store_id"],
                             "qty": qty, "value": qty * n["avg_price"]})
                avail[src] -= qty
                remaining -= qty
                if remaining <= 0:
                    break
    return pd.DataFrame(rows, columns=cols)


def recommend(lines: pd.DataFrame, transfers: pd.DataFrame, lead: int, over_days: float, watch_days: float) -> pd.DataFrame:
    """One prioritised action per store-product line, with a plain-language reason."""
    d = lines.copy().reset_index(drop=True)
    d["transfer_in"], d["transfer_out"] = 0, 0
    d["from_stores"], d["to_stores"] = "", ""
    if not transfers.empty:
        idx = pd.MultiIndex.from_frame(d[["store_id", "product_id"]])
        tin = transfers.groupby(["to_store", "product_id"]).agg(
            qty=("qty", "sum"), who=("from_store", lambda s: ", ".join(s)))
        tout = transfers.groupby(["from_store", "product_id"]).agg(
            qty=("qty", "sum"), who=("to_store", lambda s: ", ".join(s)))
        tin.index.names = tout.index.names = ["store_id", "product_id"]
        d["transfer_in"] = tin["qty"].reindex(idx).fillna(0).astype(int).to_numpy()
        d["from_stores"] = tin["who"].reindex(idx).fillna("").to_numpy()
        d["transfer_out"] = tout["qty"].reindex(idx).fillna(0).astype(int).to_numpy()
        d["to_stores"] = tout["who"].reindex(idx).fillna("").to_numpy()

    d["net_order"] = (d["order_qty"] - d["transfer_in"]).clip(lower=0)
    d["net_order_value"] = d["net_order"] * d["avg_price"]
    d["days_to_reorder"] = ((d["inventory_level"] - d["reorder_point"]) / d["mu"]).clip(lower=0)
    d["excess_share"] = d["excess_units"] / d["target_level"]

    under = d["status"] == "Understock risk"
    over = d["status"] == "Overstock"
    healthy = d["status"] == "Healthy"
    critical = under & (d["days_of_cover"] < 0.5 * lead)
    clear = over & (d["excess_share"] >= CLEAR_EXCESS_SHARE)

    d["action"] = np.select(
        [under & (d["transfer_in"] > 0) & (d["net_order"] == 0),
         under & (d["transfer_in"] > 0),
         under,
         over & (d["transfer_out"] > 0),
         clear, over,
         healthy & (d["days_to_reorder"] <= watch_days)],
        ["Transfer in", "Transfer in + order", "Order now", "Transfer out",
         "Clear excess", "Pause ordering", "Reorder soon"],
        default="No action",
    )
    d["priority"] = np.select(
        [critical, under, over & ((d["transfer_out"] > 0) | clear),
         over | (d["action"] == "Reorder soon")],
        ["Critical", "High", "Medium", "Low"], default="None",
    )
    d["why"] = d.apply(_reason, axis=1, lead=lead, over_days=over_days)
    d["transfer"] = np.where(
        d["transfer_in"] > 0, "In " + d["transfer_in"].astype(str) + " from " + d["from_stores"],
        np.where(d["transfer_out"] > 0, "Out " + d["transfer_out"].astype(str) + " to " + d["to_stores"], ""),
    )
    d["urgency"] = d["inventory_level"] / d["reorder_point"]
    d["rank"] = d["priority"].map({p: i for i, p in enumerate(PRIORITY_ORDER)})
    return d.sort_values(["rank", "urgency"]).reset_index(drop=True)


def _reason(r: pd.Series, lead: int, over_days: float) -> str:
    a = r["action"]
    inv, rop, tgt = f"{r['inventory_level']:,.0f}", f"{r['reorder_point']:,.0f}", f"{r['target_level']:,.0f}"
    if a == "Order now" and r["priority"] == "Critical":
        return (f"Stock of {inv} covers only {r['days_of_cover']:.1f} days against a {lead}-day lead time. "
                f"Order {r['net_order']:,.0f} units to reach {tgt}.")
    if a == "Order now":
        return f"Stock of {inv} is at or below the reorder point of {rop}. Order {r['net_order']:,.0f} units to reach {tgt}."
    if a == "Transfer in":
        return (f"Stock of {inv} is at or below the reorder point of {rop}. Receive {r['transfer_in']:,} units from "
                f"{r['from_stores']}, which hold surplus, so no new order is needed.")
    if a == "Transfer in + order":
        return (f"Stock of {inv} is at or below the reorder point of {rop}. Receive {r['transfer_in']:,} units from "
                f"{r['from_stores']} and order the remaining {r['net_order']:,.0f}.")
    if a == "Transfer out":
        return (f"Holds {r['excess_units']:,.0f} units above target. Send {r['transfer_out']:,} to {r['to_stores']}, "
                f"which sits at or below its reorder point.")
    if a == "Clear excess":
        return (f"Holds {r['excess_units']:,.0f} units above target ({r['excess_share']:.0%} over), worth about "
                f"{money(r['excess_value'])}. Consider a promotion or markdown.")
    if a == "Pause ordering":
        return (f"Covers {r['days_of_cover']:.1f} days, above the {over_days:g}-day limit. Hold orders; stock "
                f"reaches the reorder point in about {r['days_to_reorder']:.0f} days.")
    if a == "Reorder soon":
        if r["days_to_reorder"] < 0.1:
            return "Stock is only just above the reorder point. Prepare the next order."
        return f"Stock reaches the reorder point in about {r['days_to_reorder']:.1f} days. Prepare the next order."
    return "Stock is within its healthy range."


# ─────────────────────────────────────────────
# 3. UI HELPERS
# ─────────────────────────────────────────────
def section(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str, str]]) -> None:
    cells = "".join(
        f'<div class="kpi"><div class="k-label">{label}</div>'
        f'<div class="k-value {tone}">{value}</div><div class="k-note">{note}</div></div>'
        for label, value, note, tone in items
    )
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=(96 if legend else 58) if title else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=INK, size=13),
        title=dict(
            text=title, x=0, xanchor="left", y=0.97, yanchor="top",
            font=dict(family="Archivo, IBM Plex Sans, sans-serif", size=17),
        ) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family="IBM Plex Sans, system-ui, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


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


def tint_priority(value: str) -> str:
    bg, fg = PRIORITY_TINT.get(value, ("", ""))
    return f"background-color: {bg}; color: {fg}; font-weight: 600;" if bg else ""


def styled(table: pd.DataFrame, formats: dict, tint_col: str | None = None):
    """Number formats plus priority tint; falls back to the plain table if Styler is unavailable."""
    try:
        s = table.style.format(formats)
        return s.map(tint_priority, subset=[tint_col]) if tint_col else s
    except (ImportError, AttributeError):  # Styler needs jinja2; .map needs pandas 2.1+
        return table


# ─────────────────────────────────────────────
# 4. LOAD + SIDEBAR
# ─────────────────────────────────────────────
base = load_data()
if base.empty:
    st.error(
        f"No data found. Add `{PROCESSED_FILE.relative_to(BASE_DIR)}` "
        f"(or the raw file at `{RAW_FILE.relative_to(BASE_DIR)}`) next to app.py."
    )
    st.stop()
missing = [c for c in REQUIRED_COLUMNS if c not in base.columns]
if missing:
    st.error(f"The data is missing required columns: {', '.join(missing)}.")
    st.stop()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="name">Inventory Control</div>'
        '<div class="tag">Replenishment planner</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Planning assumptions", expanded=True):
        st.caption("The dataset has no supplier lead times, so these are inputs you set.")
        lead = st.slider("Supplier lead time (days)", 1, 7, 1)
        review = st.slider("Days between orders", 1, 7, 1)
        service = st.select_slider(
            "Service level", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
            format_func=lambda v: f"{v:.1%}",
            help="Chance of not running out during the lead time. Higher means more safety stock.",
        )
        lookback = st.slider(
            "Demand look-back (days)", 14, 90, 30,
            help="How many recent days are averaged to estimate expected demand.",
        )
        over_days = st.slider(
            "Overstock above (days of cover)", 2.0, 10.0, 3.0, step=0.5,
            help="A line is overstocked when stock lasts longer than this and sits above its target level.",
        )

    with st.expander("Recommendation rules", expanded=False):
        watch_days = st.slider(
            "Flag “reorder soon” within (days)", 0.5, 3.0, 0.5, step=0.5,
            help="Healthy lines expected to hit their reorder point within this many days get a heads-up.",
        )
        use_transfers = st.toggle(
            "Suggest store-to-store transfers", value=True,
            help="Moves surplus stock to stores that are short on the same product before ordering new stock.",
        )

    with st.expander("Scope", expanded=True):
        all_stores = sorted(base["store_id"].unique())
        stores = st.multiselect("Stores", all_stores, default=all_stores)
        min_d, max_d = base["date"].min().date(), base["date"].max().date()
        picked = st.date_input(
            "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
            help="Recommendations are made for the last day in this range.",
        )

    with st.expander("History filters", expanded=False):
        st.caption(
            "Used on the Demand and Stock health tabs. Category and region change from row to row "
            "in this dataset, so recommendations ignore them."
        )
        all_cats = sorted(base["category"].unique())
        all_regions = sorted(base["region"].unique())
        cats = st.multiselect("Category", all_cats, default=all_cats)
        regions = st.multiselect("Region", all_regions, default=all_regions)

if not stores:
    st.warning("Select at least one store in the sidebar.")
    st.stop()
if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
    st.info("Pick an end date to finish the date range.")
    st.stop()
start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

plan = apply_policy(demand_stats(lookback), lead, review, service, over_days)
scope = plan[plan["store_id"].isin(stores) & plan["date"].between(start, end)]
if scope.empty:
    st.warning(
        f"No plan is available for that range. Each line needs {MIN_HISTORY_DAYS} days of history first, "
        f"so plans start on {(base['date'].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)).date()}."
    )
    st.stop()

asof_date = scope["date"].max()
lines = scope[scope["date"] == asof_date]
hist = scope[scope["category"].isin(cats) & scope["region"].isin(regions)]
transfers = plan_transfers(lines) if use_transfers else plan_transfers(lines.iloc[0:0])
recs = recommend(lines, transfers, lead, over_days, watch_days)

# ─────────────────────────────────────────────
# 5. PAGE
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="page-head">
      <div>
        <h1>Smart Inventory Replenishment</h1>
        <div class="sub">{len(lines)} store-product lines across {len(stores)} stores, planned with a
        {lead}-day lead time and a {service:.1%} service level.</div>
      </div>
      <div class="asof">
        <div class="label">Plan date</div>
        <div class="date">{asof_date:%d %b %Y}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_over, tab_rec, tab_demand, tab_health = st.tabs(
    ["Overview", "Recommendations", "Demand and forecast", "Stock health"]
)

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

# ---------- Overview ----------
with tab_over:
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

# ---------- Recommendations ----------
with tab_rec:
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

# ---------- Demand and forecast ----------
with tab_demand:
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

# ---------- Stock health ----------
with tab_health:
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

st.caption(
    f"Data: {(PROCESSED_FILE if PROCESSED_FILE.exists() else RAW_FILE).name}, {len(base):,} rows from "
    f"{base['date'].min():%d %b %Y} to {base['date'].max():%d %b %Y}. "
    "Recommendations are rule-based and follow the planning assumptions in the sidebar."
)