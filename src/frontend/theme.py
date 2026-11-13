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

        /* Banner */
        .banner {{
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem 2rem;
            padding: 1.3rem 1.6rem; margin-bottom: 1rem; border-radius: 14px; color: #fff;
            background: linear-gradient(120deg, #1C2F4F 0%, {BLUE} 55%, #3F6FA8 100%);
            box-shadow: 0 6px 20px rgba(43,76,126,0.22);
        }}
        .banner h1 {{ font-size: 1.7rem; font-weight: 700; line-height: 1.1; margin: 0; padding: 0; color: #fff; }}
        .banner .sub {{ opacity: 0.85; margin-top: 0.3rem; font-size: 0.95rem; }}
        .banner .right {{ display: flex; gap: 0.7rem; flex-wrap: wrap; }}
        .banner .pill {{
            background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
            border-radius: 12px; padding: 0.45rem 0.9rem; text-align: left; min-width: 7.5rem;
        }}
        .banner .pill .l {{ font-size: 0.75rem; opacity: 0.8; }}
        .banner .pill .v {{ font-family: 'Archivo', sans-serif; font-weight: 600; font-size: 1.05rem; }}

        /* Tabs */
        button[data-baseweb="tab"] p {{ font-size: 1.02rem; font-weight: 500; }}
        [data-baseweb="tab-list"] {{ gap: 1.6rem; }}

        /* KPI cards */
        .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem; margin: 0.2rem 0 1rem 0; }}
        .kpi {{
            background: #fff; border: 1px solid {GRID}; border-top: 4px solid var(--tone, {BLUE});
            border-radius: 12px; padding: 0.8rem 1rem; transition: transform .15s ease, box-shadow .15s ease;
        }}
        .kpi:hover {{ transform: translateY(-3px); box-shadow: 0 8px 18px rgba(28,43,58,0.10); }}
        .k-label {{ color: {MUTED}; font-size: 0.82rem; }}
        .k-value {{
            font-family: 'Archivo', sans-serif; font-size: 1.85rem; font-weight: 700; line-height: 1.15;
            font-variant-numeric: tabular-nums; color: var(--tone, {INK});
        }}
        .k-note {{ color: {MUTED}; font-size: 0.8rem; }}

        /* Section titles */
        .section-title {{ font-family: 'Archivo', sans-serif; font-size: 1.05rem; font-weight: 600; margin: 0.8rem 0 0.2rem 0; }}
        .section-note {{ color: {MUTED}; font-size: 0.88rem; margin-bottom: 0.4rem; }}

        /* Line inspector */
        .insp {{ background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: 0.6rem;
                 box-shadow: 0 2px 10px rgba(28,43,58,0.05); }}
        .insp .title {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.15rem; }}
        .insp .sub {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 0.5rem; }}
        .chip {{ display: inline-block; padding: 0.12rem 0.65rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin-right: 0.4rem; }}
        .probbar {{ display: flex; height: 26px; border-radius: 8px; overflow: hidden; margin: 0.55rem 0 0.3rem 0; }}
        .probbar span {{ display: flex; align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 600; }}
        .problabels {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.75rem; }}
        .why {{ margin-top: 0.7rem; padding: 0.55rem 0.75rem; background: #F5F7F9; border-radius: 8px; font-size: 0.92rem; }}

        /* Cards for models */
        .mcard {{ background: #fff; border: 1px solid {GRID}; border-radius: 12px; padding: 0.8rem 1rem; height: 100%;
                  transition: transform .15s ease, box-shadow .15s ease; }}
        .mcard:hover {{ transform: translateY(-3px); box-shadow: 0 8px 18px rgba(28,43,58,0.10); }}
        .mcard.best {{ border: 2px solid {BLUE}; background: #F3F7FC; }}
        .mcard .n {{ font-family: 'Archivo', sans-serif; font-weight: 600; }}
        .mcard .f {{ font-family: 'Archivo', sans-serif; font-size: 1.7rem; font-weight: 700; color: {BLUE}; }}
        .mcard .m {{ color: {MUTED}; font-size: 0.82rem; }}
        .badge {{ background: {BLUE}; color: #fff; font-size: 0.7rem; font-weight: 600; padding: 0.1rem 0.5rem;
                  border-radius: 999px; margin-left: 0.4rem; vertical-align: 0.1rem; }}

        /* Metrics, sidebar */
        [data-testid="stMetricValue"] {{ font-family: 'Archivo', sans-serif; font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
        [data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
        .brand {{ padding: 0.2rem 0 0.6rem 0; }}
        .brand .name {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.15rem; }}
        .brand .tag {{ color: {MUTED}; font-size: 0.85rem; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] details {{ border-color: {GRID}; border-radius: 10px; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{ font-family: 'Archivo', sans-serif; font-weight: 600; }}

        @media (max-width: 640px) {{
            .block-container {{ padding-top: 3.6rem; }}
            .banner h1 {{ font-size: 1.4rem; }}
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


