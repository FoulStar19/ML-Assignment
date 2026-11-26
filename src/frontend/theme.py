"""Look and feel: page CSS, Plotly styling and small Streamlit building blocks.

Motion is kept gentle: one entrance on the header stripe and KPI cards, bars that fill when a line is
selected, and short fades when switching tabs. Everything respects the browser's reduced-motion setting.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import BG, BLUE, CURRENCY, GRID, INK, MUTED, PALETTE, PRIORITY_TINT, STATUS_COLORS, VIOLET

FONT = "Plus Jakarta Sans, system-ui, sans-serif"

# Simple outline icons (24x24 grid), drawn inline so no icon font or image is needed.
ICONS = {
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "box": '<path d="M21 8l-9-5-9 5v8l9 5 9-5V8z"/><path d="M3.3 7.5 12 12.5l8.7-5M12 22V12.5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.8 2.8L16 9.5"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/><path d="m3 17.5 9 5 9-5"/>',
    "cart": '<circle cx="9" cy="20" r="1.5"/><circle cx="18" cy="20" r="1.5"/><path d="M2 3h3l2.6 12.4a2 2 0 0 0 2 1.6h7.6a2 2 0 0 0 2-1.5L21 8H6"/>',
    "truck": '<path d="M2 6h11v10H2zM13 9h4l3 3v4h-7z"/><circle cx="6.5" cy="17.5" r="1.7"/><circle cx="16.5" cy="17.5" r="1.7"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01"/>',
}


def icon(name: str, size: int = 22) -> str:
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


def mix(hex_colour: str, amount: float) -> str:
    """Blend a colour with white (amount = share of white) for soft backgrounds."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % tuple(round(c + (255 - c) * amount) for c in (r, g, b))


def inject_css() -> None:
    root = (f":root{{--ink:{INK};--muted:{MUTED};--blue:{BLUE};--grid:{GRID};--bg:{BG};--violet:{VIOLET};"
            f"--red:{STATUS_COLORS['Understock risk']};--green:{STATUS_COLORS['Healthy']};--amber:{STATUS_COLORS['Overstock']};"
            f"--shadow:0 1px 2px rgba(30,42,68,.05),0 8px 22px rgba(30,42,68,.08);--card-shadow:0 1px 2px rgba(30,42,68,.04),0 4px 14px rgba(30,42,68,.05)}}")
    st.markdown(f"<style>{root}{_CSS}</style>", unsafe_allow_html=True)


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }

.stApp, .stApp :is(p, h1, h2, h3, h4, label, li, a, input, textarea, button, td, th),
.stApp span:not([data-testid="stIconMaterial"]):not([class*="material"]) { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.4rem; max-width: 1380px; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ───────── Header ───────── */
.hero { background: var(--ink); color: #fff; border-radius: 20px; padding: 1.15rem 1.5rem 1rem; margin-bottom: 1.1rem;
        box-shadow: var(--shadow); }
.hero-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
.logo { flex: none; width: 50px; height: 50px; border-radius: 15px; background: #fff; color: var(--blue); display: grid; place-items: center; }
.hero h1 { font-size: 1.6rem; font-weight: 800; line-height: 1.15; margin: 0; padding: 0; color: #fff; letter-spacing: -0.015em; }
.hero .sub { color: rgba(255,255,255,.72); font-size: .9rem; margin-top: .15rem; }
.hero .tags { margin-left: auto; display: flex; gap: .5rem; flex-wrap: wrap; }
.tag { background: rgba(255,255,255,.12); color: #fff; border-radius: 999px; padding: .28rem .8rem; font-size: .8rem; font-weight: 600; }
.shelf { display: flex; gap: 4px; height: 10px; margin-top: 1rem; }
.shelf span { border-radius: 999px; transform-origin: left center; animation: grow 1s cubic-bezier(.2,.7,.2,1) both; }
.shelf-legend { display: flex; gap: 1.2rem; flex-wrap: wrap; margin-top: .55rem; font-size: .8rem; color: rgba(255,255,255,.8); }
.shelf-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .4rem; }

/* ───────── Tabs (new and older Streamlit markup) ───────── */
[role="tablist"], [data-baseweb="tab-list"] { gap: .45rem; border-bottom: none !important; box-shadow: none !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .react-aria-SelectionIndicator { display: none !important; }
[data-testid="stTab"], button[data-baseweb="tab"] { height: auto; padding: .5rem 1.15rem; border-radius: 999px; background: transparent;
                             transition: background .2s ease, box-shadow .2s ease; }
[data-testid="stTab"]:hover, button[data-baseweb="tab"]:hover { background: #E3E9FA; }
[data-testid="stTab"] p, button[data-baseweb="tab"] p { font-size: .98rem; font-weight: 600; color: var(--muted); transition: color .2s ease; }
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] { background: var(--blue); box-shadow: 0 4px 12px rgba(59,91,219,.28); }
[data-testid="stTab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p { color: #fff; }
[data-testid="stTabPanel"], [data-baseweb="tab-panel"] { padding-top: 1rem; animation: fade .35s ease both; }

/* ───────── KPI cards ───────── */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin: .2rem 0 1.1rem 0; }
.kpi { display: flex; align-items: center; gap: .9rem; padding: .95rem 1.05rem; border-radius: 18px; background: var(--tint);
       border: 1px solid var(--edge); transition: transform .22s ease, box-shadow .22s ease;
       animation: rise .55s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 80ms); }
.kpi:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.k-icon { flex: none; width: 48px; height: 48px; border-radius: 15px; background: var(--tone); color: #fff; display: grid; place-items: center; }
.k-label { color: var(--muted); font-size: .82rem; font-weight: 600; }
.k-value { font-size: 1.95rem; font-weight: 800; line-height: 1.12; color: var(--tone); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.k-note { color: var(--muted); font-size: .8rem; min-height: 1.15em; display: flex; align-items: center; gap: .4rem; }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--red); animation: pulse 1.8s ease-out infinite; }

/* count-up: the number runs from 0 to its value (a plain number is shown if animation is off) */
.count { --n: var(--to); counter-reset: n var(--n); animation: count 1.1s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 80ms + 150ms); }
.count::after { content: counter(n); }
.count.tenths { counter-reset: w round(down, calc(var(--n) / 10), 1) t mod(var(--n), 10); }
.count.tenths::after { content: counter(w) "." counter(t); }
.static { display: none; }
@supports not (width: round(down, 10px, 1px)) { .count.tenths { display: none; } .static { display: inline; } }

/* ───────── Section titles, cards, chips ───────── */
.section-title { font-size: 1.05rem; font-weight: 700; margin: 1rem 0 .25rem 0; color: var(--ink); }
.section-note { color: var(--muted); font-size: .85rem; margin-bottom: .4rem; }
.insp { position: relative; background: #fff; border: 1px solid var(--grid); border-radius: 18px; padding: 1rem 1.1rem 1rem;
        margin-bottom: .8rem; box-shadow: var(--card-shadow); overflow: hidden; animation: fade .35s ease both; }
.insp::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: var(--tone); }
.insp .title { font-weight: 800; font-size: 1.25rem; letter-spacing: -0.01em; margin-top: .15rem; }
.insp .sub { margin: .4rem 0 .5rem 0; }
.chip { display: inline-block; padding: .16rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 700; margin-right: .4rem; }
.probbar { display: flex; height: 24px; border-radius: 8px; overflow: hidden; margin: .6rem 0 .3rem 0; gap: 2px; }
.probbar span { display: flex; align-items: center; justify-content: center; font-size: .75rem; font-weight: 700; animation: fillw .9s cubic-bezier(.2,.7,.2,1) both; }
.problabels { display: flex; justify-content: space-between; color: var(--muted); font-size: .72rem; font-weight: 600; }
.why { margin-top: .65rem; color: var(--muted); font-size: .9rem; }

/* stat tiles in the inspector */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: .55rem; margin: 0 0 .7rem 0; }
.stat { background: #fff; border: 1px solid var(--grid); border-radius: 14px; padding: .55rem .7rem; box-shadow: var(--card-shadow);
        border-bottom: 3px solid var(--c); transition: transform .2s ease; }
.stat:hover { transform: translateY(-2px); }
.stat .l { color: var(--muted); font-size: .72rem; font-weight: 600; }
.stat .v { font-size: 1.15rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.after { display: flex; align-items: center; justify-content: space-between; gap: .6rem; background: #fff; border: 1px solid var(--grid);
         border-radius: 14px; padding: .55rem .9rem; margin: .1rem 0 .7rem 0; box-shadow: var(--card-shadow); }
.after .l { color: var(--muted); font-size: .78rem; font-weight: 600; }
.after .v { font-size: 1.3rem; font-weight: 800; font-variant-numeric: tabular-nums; }

/* stock meter */
.meter { background: #fff; border: 1px solid var(--grid); border-radius: 18px; padding: .9rem 1.1rem .6rem; box-shadow: var(--card-shadow); margin-bottom: .6rem; }
.meter .mt { font-weight: 700; font-size: .95rem; margin-bottom: .55rem; display: flex; justify-content: space-between; gap: .5rem; flex-wrap: wrap; }
.meter .mt small { color: var(--muted); font-weight: 600; font-size: .78rem; }
.track { position: relative; height: 26px; border-radius: 9px; overflow: hidden; display: flex; }
.zone { height: 100%; }
.fill { position: absolute; left: 0; top: 7px; height: 12px; border-radius: 0 6px 6px 0; background: var(--ink); transform-origin: left center; animation: grow .9s cubic-bezier(.2,.7,.2,1) both; }
.add { position: absolute; top: 7px; height: 12px; border-radius: 6px; transform-origin: left center; animation: grow .9s .15s cubic-bezier(.2,.7,.2,1) both;
       background: repeating-linear-gradient(135deg, var(--blue) 0 5px, #8FA4F0 5px 10px); }
.ticks { position: relative; height: 44px; margin-top: 2px; }
.tick { position: absolute; top: 0; transform: translateX(-50%); text-align: center; font-size: .7rem; font-weight: 600; color: var(--muted); white-space: nowrap; line-height: 1.15; }
.tick::before { content: ""; display: block; width: 2px; height: 7px; margin: 0 auto 2px; background: var(--c); border-radius: 2px; }
.tick b { color: var(--ink); margin-left: .25rem; font-size: .74rem; }
.tick.low { top: 22px; }
.tick.low::before { height: 3px; }

/* model cards */
.mcard { position: relative; background: #fff; border: 1px solid var(--grid); border-radius: 18px; padding: .85rem 1rem; height: 100%;
         box-shadow: var(--card-shadow); border-top: 5px solid var(--c); transition: transform .22s ease, box-shadow .22s ease; }
.mcard:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.mcard.best { background: linear-gradient(180deg, var(--tint), #fff 70%); }
.mcard .n { font-weight: 700; font-size: .95rem; }
.mcard .f { font-size: 1.7rem; font-weight: 800; color: var(--c); letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.mcard .m { color: var(--muted); font-size: .8rem; }
.mcard .bar { height: 6px; background: var(--grid); border-radius: 99px; margin-top: .5rem; overflow: hidden; }
.mcard .bar i { display: block; height: 100%; border-radius: 99px; background: var(--c); transform-origin: left center; animation: grow 1s cubic-bezier(.2,.7,.2,1) both; }
.badge { background: var(--blue); color: #fff; font-size: .68rem; font-weight: 700; padding: .12rem .55rem; border-radius: 999px; margin-left: .45rem; vertical-align: .1rem; }

/* native widgets */
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--grid); border-radius: 14px; padding: .6rem .85rem; box-shadow: var(--card-shadow); }
[data-testid="stMetricValue"] { font-weight: 800; font-size: 1.35rem; font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .8rem; font-weight: 600; }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid var(--grid); border-radius: 18px; padding: 0; overflow: hidden; box-shadow: var(--card-shadow); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; box-shadow: var(--card-shadow); }
[data-testid="stExpander"] details { border-radius: 14px; border-color: var(--grid); background: #fff; }
[data-testid="stExpander"] summary p { font-weight: 700; }
button[kind="secondary"], [data-testid="stDownloadButton"] button { border-radius: 999px; transition: transform .15s ease, box-shadow .15s ease; }
[data-testid="stDownloadButton"] button:hover { transform: translateY(-1px); box-shadow: var(--card-shadow); }
[data-testid="stButtonGroup"] button, [data-testid="stSegmentedControl"] button, [data-testid="stPills"] button { transition: background .2s ease, transform .15s ease; }
[data-testid="stButtonGroup"] button:hover, [data-testid="stSegmentedControl"] button:hover, [data-testid="stPills"] button:hover { transform: translateY(-1px); }
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button { padding: .25rem .55rem; font-size: .82rem; }

/* sidebar */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child { max-height: 6.2rem; overflow-y: auto; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--grid); }
.brand { display: flex; align-items: center; gap: .7rem; padding: .2rem 0 .8rem 0; }
.brand .logo { width: 40px; height: 40px; border-radius: 12px; background: var(--blue); color: #fff; }
.brand b { font-size: 1.05rem; font-weight: 800; display: block; line-height: 1.15; }
.brand small { color: var(--muted); font-size: .75rem; }

@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes fillw { from { width: 0; opacity: .4; } }
@keyframes count { from { --n: 0; } to { --n: var(--to); } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(229,72,77,.55); } 70% { box-shadow: 0 0 0 9px rgba(229,72,77,0); } 100% { box-shadow: 0 0 0 0 rgba(229,72,77,0); } }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
@media (max-width: 640px) {
    .block-container { padding-top: 3.6rem; }
    .hero h1 { font-size: 1.3rem; }
    .hero .tags { margin-left: 0; }
    .stats { grid-template-columns: 1fr 1fr; }
}
"""


def money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"


def section(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header and KPI cards
# ─────────────────────────────────────────────
def hero(asof: str, models: list[str], n_under: int, n_healthy: int, n_over: int) -> None:
    """Title band. The stripe along the bottom is the stock mix: red understock, green healthy, amber overstock."""
    parts = [("Understock risk", n_under), ("Healthy", n_healthy), ("Overstock", n_over)]
    total = max(sum(n for _, n in parts), 1)
    stripe = "".join(
        f'<span style="flex:{max(n, 0.0001) / total:.5f};background:{STATUS_COLORS[name]};animation-delay:{i * 120}ms"></span>'
        for i, (name, n) in enumerate(parts))
    legend = "".join(f'<span><i style="background:{STATUS_COLORS[name]}"></i>{n} {name.lower()}</span>' for name, n in parts)
    tags = "".join(f'<span class="tag">{t}</span>' for t in [asof] + models)
    st.markdown(
        f'<div class="hero"><div class="hero-row"><div class="logo">{icon("box", 28)}</div>'
        f'<div><h1>Smart Inventory Replenishment</h1><div class="sub">What to order, move or hold today</div></div>'
        f'<div class="tags">{tags}</div></div>'
        f'<div class="shelf">{stripe}</div><div class="shelf-legend">{legend}</div></div>',
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown(f'<div class="brand"><div class="logo">{icon("box", 22)}</div>'
                f'<div><b>Inventory Control</b><small>Planner settings</small></div></div>', unsafe_allow_html=True)


def _count_html(value: float, fmt: str) -> str:
    """The number, counting up from 0. Money is shortened (457.7K) so the count-up stays a plain number."""
    if fmt == "money":
        v = abs(value)
        for scale, suffix in ((1e6, "M"), (1e3, "K")):
            if v >= scale:
                tenths = round(v / scale * 10)
                return (f'<span class="count tenths" style="--to:{tenths}"></span><span class="static">{tenths / 10:.1f}</span>{suffix}')
        value = v
    return f'<span class="count" style="--to:{int(round(value))}"></span>'


def kpi_row(items: list[dict]) -> None:
    """items: dicts with label, value, note, tone (red/green/amber/blue), icon, fmt (int/money), pulse (optional)."""
    tones = {"red": STATUS_COLORS["Understock risk"], "green": STATUS_COLORS["Healthy"],
             "amber": STATUS_COLORS["Overstock"], "blue": BLUE}
    cells = []
    for i, it in enumerate(items):
        c = tones[it["tone"]]
        dot = '<span class="pulse"></span>' if it.get("pulse") else ""
        cells.append(
            f'<div class="kpi" style="--tone:{c};--tint:{mix(c, .9)};--edge:{mix(c, .72)};--i:{i}">'
            f'<div class="k-icon">{icon(it["icon"])}</div><div><div class="k-label">{it["label"]}</div>'
            f'<div class="k-value">{CURRENCY if it.get("fmt") == "money" else ""}{_count_html(it["value"], it.get("fmt", "int"))}</div>'
            f'<div class="k-note">{dot}{it["note"]}</div></div></div>')
    st.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def stat_row(items: list[tuple[str, str, str]]) -> None:
    """Small tiles: (label, value, colour)."""
    cells = "".join(f'<div class="stat" style="--c:{c}"><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v, c in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


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


def stock_meter(title: str, on_hand: float, order: float, reorder: float, target: float, overstock: float) -> str:
    """A shelf gauge: stock on hand (and the order on top of it) against the reorder, target and overstock levels."""
    top = max(overstock, target, reorder, on_hand + order) * 1.08 or 1
    pct = lambda v: max(0.0, min(100.0, v / top * 100))
    r, o = pct(reorder), pct(overstock)
    zones = (f'<div class="zone" style="width:{r:.2f}%;background:{mix(STATUS_COLORS["Understock risk"], .78)}"></div>'
             f'<div class="zone" style="width:{max(o - r, 0):.2f}%;background:{mix(STATUS_COLORS["Healthy"], .78)}"></div>'
             f'<div class="zone" style="flex:1;background:{mix(STATUS_COLORS["Overstock"], .72)}"></div>')
    fill = f'<div class="fill" style="width:{pct(on_hand):.2f}%"></div>'
    add = (f'<div class="add" style="left:{pct(on_hand):.2f}%;width:{max(pct(on_hand + order) - pct(on_hand), 0):.2f}%"></div>'
           if order > 0 else "")
    marks = sorted([("Reorder", reorder, STATUS_COLORS["Understock risk"]), ("Target", target, BLUE),
                    ("Overstock", overstock, STATUS_COLORS["Overstock"])], key=lambda m: m[1])
    ticks, last, low = [], -100.0, False
    for name, value, colour in marks:
        x = pct(value)
        low = (not low) if x - last < 22 else False          # labels that would collide drop to a second row
        ticks.append(f'<div class="tick{" low" if low else ""}" style="left:{x:.2f}%;--c:{colour}">{name}<b>{value:,.0f}</b></div>')
        last = x
    sub = f"<small>On hand {on_hand:,.0f}" + (f" + order {order:,.0f}" if order > 0 else "") + "</small>"
    return (f'<div class="meter"><div class="mt"><span>{title}</span>{sub}</div>'
            f'<div class="track">{zones}{fill}{add}</div><div class="ticks">{"".join(ticks)}</div></div>')


def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=(96 if legend else 62) if title else 28, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        font=dict(family=FONT, color=INK, size=13),
        title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top", font=dict(family=FONT, size=17)) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GRID),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=key)


def tint_priority(value: str) -> str:
    bg, fg = PRIORITY_TINT.get(value, ("", ""))
    return f"background-color: {bg}; color: {fg}; font-weight: 700;" if bg else ""


def styled(table: pd.DataFrame, formats: dict, tint_col: str | None = None):
    """Number formats plus priority tint; falls back to the plain table if Styler is unavailable."""
    try:
        s = table.style.format(formats)
        return s.map(tint_priority, subset=[tint_col]) if tint_col else s
    except (ImportError, AttributeError):  # Styler needs jinja2; .map needs pandas 2.1+
        return table
