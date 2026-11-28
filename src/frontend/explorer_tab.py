"""Tab 1, Data explorer: what every column means, how it relates to Units_Sold, the Promotion / Holiday count-model check,
and how Demand_Level is derived."""
import numpy as np
import pandas as pd
import streamlit as st

from . import charts, columns, data, flag_effects
from .config import DATE_COL, TARGET
from .ui import STRETCH

LEVELS = ["Low", "Medium", "High", "Very High"]


def prepare(raw):
    d = raw.copy()
    for col in ("Promotion", "Holiday"):                     # keep Yes/No as text for the explorer
        if col in d:
            d[col] = d[col].astype(str)
    price = d["Unit_Price"] * (1 - d["Discount_Percent"] / 100)
    d["Log_Effective_Price"] = np.log(price.where(price > 0))
    d["Price_Gap"] = d["Unit_Price"] / d["Competitor_Price"].where(d["Competitor_Price"] > 0)
    return d


def kind(s, col):
    if s.nunique() == len(s):
        return "id"
    if col == DATE_COL:
        return "date"
    return "numeric" if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15 else "category"


def group_means(d, col):
    g = d.groupby(col, observed=True)[TARGET].agg(Mean_Units="mean", Records="size").reset_index()
    return g


# ─────────────────────────────────────────────
# Column dictionary and per-column explorer
# ─────────────────────────────────────────────
def dictionary_card(d):
    with st.container(border=True):
        st.markdown("##### Column dictionary")
        rows = columns.dictionary(d)
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=300, **STRETCH)
        st.caption("Meanings are written from the column names and the cleaning rules. Edit `columns.py` if your data source says otherwise.")


def explorer_card(d):
    with st.container(border=True):
        col = st.selectbox("Explore a column", list(d.columns), index=list(d.columns).index("Promotion") if "Promotion" in d else 0)
        s = d[col]
        k = kind(s, col)
        a, b, c, e = st.columns(4)
        a.metric("Role", columns.role(col))
        b.metric("Unique values", f"{s.nunique():,}")
        c.metric("Missing", f"{s.isna().mean():.1%}")
        if k == "numeric" and col != TARGET:
            e.metric("Correlation with Units_Sold", f"{s.corr(d[TARGET]):+.2f}")
        elif k == "category" and s.nunique() == 2 and col != TARGET:
            g = group_means(d, col).set_index(col)["Mean_Units"]
            e.metric(f"Units sold: {g.idxmax()} vs {g.idxmin()}", f"x{g.max() / g.min():.2f}")
        st.caption(columns.DESCRIPTIONS.get(col, ""))

        left, right = st.columns(2)
        if k == "id":
            st.info("This column is an identifier: every value is different, so there is no pattern to analyse.")
            return
        left.markdown(f"**Distribution of {col}**")
        right.markdown(f"**{col} vs Units_Sold**" if col != TARGET else "**Units_Sold by month**")
        if col == TARGET:
            left.altair_chart(charts.hist(d, col, "Units sold"), **STRETCH)
            m = d.assign(Month=d[DATE_COL].dt.to_period("M").dt.to_timestamp()).groupby("Month")[TARGET].mean().rename("Mean_Units").reset_index()
            right.altair_chart(charts.month_line(m), **STRETCH)
        elif k == "date":
            months = d.assign(Month=d[DATE_COL].dt.to_period("M").dt.to_timestamp()).groupby("Month")
            m = months[TARGET].mean().rename("Mean_Units").reset_index()
            left.altair_chart(charts.month_line(months.size().rename("Mean_Units").reset_index()).encode(
                y=charts.alt.Y("Mean_Units:Q", title="Records per month")), **STRETCH)
            right.altair_chart(charts.month_line(m), **STRETCH)
        elif k == "numeric":
            left.altair_chart(charts.hist(d, col, col), **STRETCH)
            binned = d[[col, TARGET]].dropna()
            q = pd.qcut(binned[col], 10, duplicates="drop")
            g = binned.groupby(q, observed=True).agg(Mid=(col, "mean"), Mean_Units=(TARGET, "mean"), Records=(TARGET, "size")).reset_index(drop=True)
            right.altair_chart(charts.trend_line(g, "Mid", f"{col} (average of each tenth of the records)"), **STRETCH)
        else:
            counts = s.astype(str).value_counts().rename_axis(col).reset_index(name="Records")
            g = group_means(d.assign(**{col: s.astype(str)}), col)
            if len(counts) > 30:
                left.caption(f"{len(counts)} values: showing the 15 most common.")
                counts = counts.head(15)
                g = g.sort_values("Mean_Units", ascending=False)
                g = pd.concat([g.head(8), g.tail(7)])
                right.caption("Highest 8 and lowest 7 average units sold.")
            left.altair_chart(charts.count_bars(counts, col), **STRETCH)
            right.altair_chart(charts.mean_by_category(g, col), **STRETCH)


# ─────────────────────────────────────────────
# Promotion and Holiday: count-model check
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting the count models...")
def _flag_models(_df, mtime):
    return flag_effects.analyze(_df)


def flags_card(raw):
    with st.container(border=True):
        st.markdown("##### Promotion and Holiday: count-model check")
        st.caption("Units_Sold and Returns are modelled from the two flags only. A zero-inflated model is only needed when there are far more "
                   "zeros than a Poisson allows. Lower AIC = better fit.")
        try:
            comp, eff = _flag_models(raw[["Promotion", "Holiday", TARGET, "Returns"]], len(raw))
        except ImportError:
            st.warning("This check needs the statsmodels package: `pip install statsmodels`.")
            return
        except Exception as e:
            st.warning(f"Could not fit the count models: {e}")
            return
        show = comp.assign(Zeros=comp["Zeros"].map("{:.0%}".format), AIC=comp["AIC"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}"),
                           Best=comp["Best"].map({True: "best fit", False: ""}))
        st.dataframe(show[["Outcome", "Model", "Zeros", "AIC", "Best", "Note"]], hide_index=True, **STRETCH)
        for out in comp["Outcome"].unique():
            best = comp[(comp["Outcome"] == out) & comp["Best"]].iloc[0]
            zero = comp[(comp["Outcome"] == out) & (comp["Model"] == "Zero-inflated Poisson")]
            nz = float(comp[comp["Outcome"] == out]["Zeros"].iloc[0])
            st.markdown(f"- **{out}** ({nz:.0%} zeros): best fit is **{best['Model']}**" +
                        ("; the zero-inflated Poisson does not beat it, so the extra variation comes from "
                         "overdispersion rather than extra zeros." if not zero.empty and not zero["Best"].iloc[0] else "."))
        pick = st.radio("Effects from", ["Negative binomial", "Zero-inflated Poisson"], horizontal=True)
        out = st.radio("Outcome", sorted(comp["Outcome"].unique()), horizontal=True)
        e = eff[(eff["Model"] == pick) & (eff["Outcome"] == out)]
        if e.empty:
            st.caption("This model could not be fitted for this outcome.")
        else:
            st.dataframe(e[["Part", "Flag", "Ratio", "Low", "High", "p-value"]].style.format(
                {"Ratio": "{:.2f}", "Low": "{:.2f}", "High": "{:.2f}", "p-value": "{:.3g}"}), hide_index=True, **STRETCH)
            st.caption("Rate ratio = how many times higher the expected count is when the flag is on (95% interval Low to High). "
                       "Odds ratio (zero part) = change in the odds of a structural zero. Units_Sold has no zeros, so it has no zero part.")


# ─────────────────────────────────────────────
# Demand_Level
# ─────────────────────────────────────────────
def demand_level_card(raw):
    with st.container(border=True):
        st.markdown("##### How Demand_Level is derived")
        if "Demand_Level" not in raw or "Sales_Amount" not in raw:
            st.info("Demand_Level and Sales_Amount are not in the raw data file.")
            return
        d = raw[["Sales_Amount", "Demand_Level"]].dropna()
        g = d.groupby("Demand_Level")["Sales_Amount"].agg(["min", "max", "size"]).reindex([l for l in LEVELS if l in set(d["Demand_Level"])])
        cuts = [round((g["max"].iloc[i] + g["min"].iloc[i + 1]) / 2) for i in range(len(g) - 1)]
        rebuilt = pd.cut(d["Sales_Amount"], [-np.inf, *cuts, np.inf], labels=list(g.index))
        match = float((rebuilt.astype(str) == d["Demand_Level"]).mean())
        left, right = st.columns([1, 1.4])
        table = pd.DataFrame({"Demand level": g.index, "Sales_Amount from": g["min"].round(0).astype(int).to_numpy(),
                              "to": g["max"].round(0).astype(int).to_numpy(), "Records": g["size"].to_numpy(),
                              "Share": (g["size"] / g["size"].sum()).map("{:.0%}".format).to_numpy()})
        left.dataframe(table, hide_index=True, **STRETCH)
        left.metric("Rebuilt from Sales_Amount alone", f"{match:.1%} match")
        right.altair_chart(charts.revenue_hist(d, cuts), **STRETCH)
        st.caption(f"Demand_Level is a set of revenue bands: the limits are about {', '.join(f'{c:,}' for c in cuts)}. Because Sales_Amount = units x price x "
                   "(1 - discount), a record can be 'High' from a high price with few units, so it is not the same as high unit demand. "
                   "It is built from the target, so it is never used as a model feature.")


def render():
    raw = data.load_raw()
    if raw is None:
        st.info("The raw data file `data/raw/retail_sales_forecast_dataset.csv` was not found, so the data explorer is not available.")
        return
    d = prepare(raw)
    dictionary_card(d)
    explorer_card(d)
    left, right = st.columns(2)
    with left:
        flags_card(raw)
    with right:
        demand_level_card(raw)
