"""Look and feel: page CSS, Plotly styling and small Streamlit building blocks."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (CURRENCY, GRID, INK, MUTED, BLUE, PRIORITY_COLORS, PRIORITY_TINT, STATUS_COLORS)


def inject_css() -> None:
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


