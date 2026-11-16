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
            font-family: 'Archivo', 'IBM Plex Sans', system-ui, sans-serif; letter-spacing: -0.01em;
        }}
        .block-container {{ padding-top: 1.6rem; max-width: 1360px; }}
        footer {{ visibility: hidden; }}

        /* Header: title and one muted line */
        .head {{ display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.2rem 1.5rem;
                 padding-bottom: 0.7rem; margin-bottom: 0.4rem; border-bottom: 1px solid {GRID}; }}
        .head h1 {{ font-size: 1.5rem; font-weight: 700; line-height: 1.2; margin: 0; padding: 0; color: {INK}; }}
        .head .meta {{ color: {MUTED}; font-size: 0.9rem; }}

        /* Tabs */
        button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 500; }}
        [data-baseweb="tab-list"] {{ gap: 1.4rem; }}

        /* KPIs: flat, value carries the colour */
        .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.7rem; margin: 0.4rem 0 0.9rem 0; }}
        .kpi {{ background: #fff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.65rem 0.9rem; }}
        .k-label {{ color: {MUTED}; font-size: 0.8rem; }}
        .k-value {{ font-family: 'Archivo', sans-serif; font-size: 1.6rem; font-weight: 700; line-height: 1.2;
                    font-variant-numeric: tabular-nums; color: var(--tone, {INK}); }}
        .k-note {{ color: {MUTED}; font-size: 0.78rem; min-height: 1.1em; }}

        /* Section titles */
        .section-title {{ font-family: 'Archivo', sans-serif; font-size: 1rem; font-weight: 600; margin: 0.8rem 0 0.2rem 0; }}
        .section-note {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 0.4rem; }}

        /* Line inspector */
        .insp {{ background: #fff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.6rem; }}
        .insp .title {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.1rem; }}
        .insp .sub {{ margin: 0.25rem 0 0.4rem 0; }}
        .chip {{ display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; }}
        .probbar {{ display: flex; height: 22px; border-radius: 6px; overflow: hidden; margin: 0.5rem 0 0.25rem 0; }}
        .probbar span {{ display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; }}
        .problabels {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.72rem; }}
        .why {{ margin-top: 0.6rem; color: {MUTED}; font-size: 0.9rem; }}

        /* Model cards */
        .mcard {{ background: #fff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.7rem 0.9rem; height: 100%; }}
        .mcard.best {{ border: 1.5px solid {BLUE}; }}
        .mcard .n {{ font-family: 'Archivo', sans-serif; font-weight: 600; font-size: 0.95rem; }}
        .mcard .f {{ font-family: 'Archivo', sans-serif; font-size: 1.5rem; font-weight: 700; color: {BLUE}; }}
        .mcard .m {{ color: {MUTED}; font-size: 0.8rem; }}
        .badge {{ background: {BLUE}; color: #fff; font-size: 0.68rem; font-weight: 600; padding: 0.1rem 0.5rem;
                  border-radius: 999px; margin-left: 0.4rem; vertical-align: 0.1rem; }}

        /* Metrics, sidebar */
        [data-testid="stMetricValue"] {{ font-family: 'Archivo', sans-serif; font-size: 1.3rem; font-variant-numeric: tabular-nums; }}
        [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.82rem; }}
        .brand {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.1rem; padding: 0.2rem 0 0.6rem 0; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] details {{ border-color: {GRID}; border-radius: 8px; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{ font-family: 'Archivo', sans-serif; font-weight: 600; }}

        @media (max-width: 640px) {{
            .block-container {{ padding-top: 3.6rem; }}
            .head h1 {{ font-size: 1.3rem; }}
            .kpis {{ grid-template-columns: 1fr 1fr; }}
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
    """items: (label, value, note, tone) where tone is bad, good, warn or empty."""
    tones = {"bad": PRIORITY_COLORS["High"], "good": STATUS_COLORS["Healthy"], "warn": STATUS_COLORS["Overstock"], "": BLUE}
    cells = "".join(
        f'<div class="kpi" style="--tone:{tones.get(tone, BLUE)}"><div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div><div class="k-note">{note}</div></div>'
        for label, value, note, tone in items
    )
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def chip(text: str, bg: str, fg: str) -> str:
    return f'<span class="chip" style="background:{bg};color:{fg}">{text}</span>'


def prob_bar(p_under: float, p_healthy: float, p_over: float) -> str:
    """Stacked bar of the model's three class probabilities."""
    parts = [(p_under, STATUS_COLORS["Understock risk"], "white"), (p_healthy, STATUS_COLORS["Healthy"], "white"),
             (p_over, STATUS_COLORS["Overstock"], INK)]
    spans = "".join(
        f'<span style="width:{p * 100:.1f}%;background:{bg};color:{fg}">{p:.0%}</span>' if p >= 0.07
        else f'<span style="width:{p * 100:.1f}%;background:{bg}"></span>'
        for p, bg, fg in parts
    )
    return (f'<div class="probbar">{spans}</div>'
            '<div class="problabels"><span>Understock</span><span>Healthy</span><span>Overstock</span></div>')


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


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=key)




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


