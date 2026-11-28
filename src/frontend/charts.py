import altair as alt
import pandas as pd

from .config import STATUS_COLORS

STATUS_SCALE = alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values()))


def status_bar(df):
    counts = df["Status"].value_counts().reindex(list(STATUS_COLORS), fill_value=0).rename_axis("Status").reset_index(name="Rows")
    return (alt.Chart(counts).mark_bar(cornerRadiusEnd=4)
            .encode(y=alt.Y("Status:N", sort=list(STATUS_COLORS), title=None), x=alt.X("Rows:Q", title="Records"),
                    color=alt.Color("Status:N", scale=STATUS_SCALE, legend=None), tooltip=["Status", "Rows"])
            .properties(height=150))


def status_by_category(df):
    return (alt.Chart(df).mark_bar()
            .encode(y=alt.Y("Product_Category:N", title=None), x=alt.X("count():Q", stack="normalize", title="Share of records", axis=alt.Axis(format="%")),
                    color=alt.Color("Status:N", scale=STATUS_SCALE, legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Product_Category", "Status", "count()"])
            .properties(height=260))


def cover_histogram(df, rules):
    rule_df = pd.DataFrame({"days": [rules.understock_days, rules.overstock_days]})
    hist = (alt.Chart(df).mark_bar(color="#6c8ebf", opacity=0.8)
            .encode(x=alt.X("Cover_Days:Q", bin=alt.Bin(maxbins=40), title="Days of cover (stock / predicted daily units)"),
                    y=alt.Y("count():Q", title="Records")))
    rules_ = alt.Chart(rule_df).mark_rule(strokeDash=[5, 4], size=2, color="#444").encode(x="days:Q")
    return (hist + rules_).properties(height=260)


def stock_vs_demand(inventory, pred, rules):
    d = pd.DataFrame({
        "Item": ["Stock on hand", f"Demand ({rules.understock_days} days)", f"Target stock ({rules.target_days} days)", f"Overstock line ({rules.overstock_days} days)"],
        "Units": [inventory, pred * rules.understock_days, pred * rules.target_days, pred * rules.overstock_days]})
    d["Units"] = d["Units"].round(0)
    return (alt.Chart(d).mark_bar(cornerRadiusEnd=4)
            .encode(y=alt.Y("Item:N", sort=None, title=None), x=alt.X("Units:Q"),
                    color=alt.condition(alt.datum.Item == "Stock on hand", alt.value("#1f77b4"), alt.value("#b0b7c3")),
                    tooltip=["Item", "Units"])
            .properties(height=180))


def what_if_line(df, xcol, xtitle):
    return (alt.Chart(df).mark_line(point=True, color="#1f77b4")
            .encode(x=alt.X(f"{xcol}:Q", title=xtitle), y=alt.Y("Predicted:Q", title="Predicted units", scale=alt.Scale(zero=False)),
                    tooltip=[xcol, alt.Tooltip("Predicted:Q", format=".1f")])
            .properties(height=200))


def metric_bars(df, metric, title, lower_better=True):
    d = df.reset_index().rename(columns={"index": "Model"})
    best = d[metric].idxmin() if lower_better else d[metric].idxmax()
    d["Best"] = d.index == best
    return (alt.Chart(d).mark_bar(cornerRadiusEnd=4)
            .encode(y=alt.Y("Model:N", sort=None, title=None), x=alt.X(f"{metric}:Q", title=title),
                    color=alt.condition("datum.Best", alt.value("#2ca02c"), alt.value("#9aa5b1")),
                    tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".3f")])
            .properties(height=160))


def actual_vs_predicted(actual, pred):
    d = pd.DataFrame({"Actual": actual, "Predicted": pred})
    top = float(max(d.max()))
    pts = alt.Chart(d).mark_circle(size=14, opacity=0.25).encode(x=alt.X("Actual:Q", title="Actual units sold"), y=alt.Y("Predicted:Q", title="Predicted units sold"))
    line = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(color="#d62728", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    return (pts + line).properties(height=300)


def residual_hist(actual, pred):
    d = pd.DataFrame({"Error": pred - actual})
    return (alt.Chart(d).mark_bar(color="#6c8ebf")
            .encode(x=alt.X("Error:Q", bin=alt.Bin(maxbins=40), title="Predicted - actual (units)"), y=alt.Y("count():Q", title="Records"))
            .properties(height=300))


def bar_h(df, label, value, title, color="#6c8ebf"):
    return (alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4)
            .encode(y=alt.Y(f"{label}:N", sort="-x", title=None), x=alt.X(f"{value}:Q", title=title),
                    tooltip=[label, alt.Tooltip(f"{value}:Q", format=".3f")])
            .properties(height=max(120, 26 * len(df))))