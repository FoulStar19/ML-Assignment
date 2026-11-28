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


# ─────────────────────────────────────────────
# Charts added for the data explorer, predict & explore and recommender tabs
# ─────────────────────────────────────────────
def hist(df, col, title, maxbins=30, color="#6c8ebf"):
    return (alt.Chart(df[[col]].dropna()).mark_bar(color=color)
            .encode(x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=maxbins), title=title), y=alt.Y("count():Q", title="Records"))
            .properties(height=240))


def count_bars(counts, label, title="Records"):
    return (alt.Chart(counts).mark_bar(color="#6c8ebf", cornerRadiusEnd=4)
            .encode(y=alt.Y(f"{label}:N", sort="-x", title=None), x=alt.X("Records:Q", title=title), tooltip=[label, "Records"])
            .properties(height=max(120, 24 * len(counts))))


def mean_by_category(g, label):
    """g: one row per group with Mean_Units and Records."""
    return (alt.Chart(g).mark_bar(color="#1f77b4", cornerRadiusEnd=4)
            .encode(y=alt.Y(f"{label}:N", sort="-x", title=None), x=alt.X("Mean_Units:Q", title="Average units sold"),
                    tooltip=[label, alt.Tooltip("Mean_Units:Q", format=".1f"), "Records"])
            .properties(height=max(120, 24 * len(g))))


def trend_line(g, xcol, xtitle):
    """g: binned values of a column (xcol) with Mean_Units and Records."""
    return (alt.Chart(g).mark_line(point=True, color="#1f77b4")
            .encode(x=alt.X(f"{xcol}:Q", title=xtitle, scale=alt.Scale(zero=False)),
                    y=alt.Y("Mean_Units:Q", title="Average units sold", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip(f"{xcol}:Q", format=".2f"), alt.Tooltip("Mean_Units:Q", format=".1f"), "Records"])
            .properties(height=240))


def month_line(g):
    return (alt.Chart(g).mark_line(color="#1f77b4")
            .encode(x=alt.X("Month:T", title=None), y=alt.Y("Mean_Units:Q", title="Average units sold", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("Month:T", format="%b %Y"), alt.Tooltip("Mean_Units:Q", format=".1f")])
            .properties(height=240))


LEVEL_COLORS = {"Low": "#9aa5b1", "Medium": "#6c8ebf", "High": "#1f77b4", "Very High": "#1c2330"}


def revenue_hist(df, cuts):
    """Sales_Amount histogram coloured by Demand_Level, with the band limits marked."""
    d = df[["Sales_Amount", "Demand_Level"]].dropna()
    hist_ = (alt.Chart(d).mark_bar()
             .encode(x=alt.X("Sales_Amount:Q", bin=alt.Bin(maxbins=60), title="Sales_Amount"), y=alt.Y("count():Q", title="Records"),
                     color=alt.Color("Demand_Level:N", scale=alt.Scale(domain=list(LEVEL_COLORS), range=list(LEVEL_COLORS.values())),
                                     legend=alt.Legend(orient="bottom", title=None))))
    rules_ = alt.Chart(pd.DataFrame({"x": cuts})).mark_rule(strokeDash=[5, 4], size=2, color="#d62728").encode(x="x:Q")
    return (hist_ + rules_).properties(height=240)


def scenario_bars(df):
    """df: Scenario, Predicted, Current (True for the scenario that matches the inputs)."""
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=4)
            .encode(y=alt.Y("Scenario:N", sort=None, title=None), x=alt.X("Predicted:Q", title="Predicted units"),
                    color=alt.condition("datum.Current", alt.value("#1f77b4"), alt.value("#b0b7c3")),
                    tooltip=["Scenario", alt.Tooltip("Predicted:Q", format=".1f")])
            .properties(height=170))


def heatmap(df, xcol, ycol, xtitle, ytitle):
    """df: xcol, ycol and Predicted. Darker = more units."""
    base = alt.Chart(df).encode(x=alt.X(f"{xcol}:O", title=xtitle), y=alt.Y(f"{ycol}:O", title=ytitle, sort="descending"))
    rect = base.mark_rect().encode(color=alt.Color("Predicted:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                   tooltip=[xcol, ycol, alt.Tooltip("Predicted:Q", format=".1f")])
    mid = float(df["Predicted"].mean())
    text = base.mark_text(fontSize=11).encode(text=alt.Text("Predicted:Q", format=".0f"),
                                              color=alt.condition(alt.datum.Predicted > mid, alt.value("white"), alt.value("#1c2330")))
    return (rect + text).properties(height=240)


def model_bars(df):
    """df: Model, Predicted, Selected."""
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=4)
            .encode(y=alt.Y("Model:N", sort=None, title=None), x=alt.X("Predicted:Q", title="Predicted units"),
                    color=alt.condition("datum.Selected", alt.value("#2ca02c"), alt.value("#9aa5b1")),
                    tooltip=["Model", alt.Tooltip("Predicted:Q", format=".1f")])
            .properties(height=170))


def stock_vs_plan(inventory, row):
    d = pd.DataFrame({"Item": ["Stock on hand", "Expected demand", "Demand + safety stock (reorder point)", "Target stock"],
                      "Units": [inventory, row["Expected_Demand"], row["Reorder_Point"], row["Target_Stock"]]})
    d["Units"] = d["Units"].round(0)
    return (alt.Chart(d).mark_bar(cornerRadiusEnd=4)
            .encode(y=alt.Y("Item:N", sort=None, title=None), x=alt.X("Units:Q"),
                    color=alt.condition(alt.datum.Item == "Stock on hand", alt.value("#1f77b4"), alt.value("#b0b7c3")),
                    tooltip=["Item", "Units"])
            .properties(height=190))


def abc_xyz_matrix(counts, selected=None):
    """counts: ABC, XYZ, Items. The selected segment (e.g. "AX") gets a dark outline."""
    d = counts.assign(Selected=(counts["ABC"] + counts["XYZ"]) == selected)
    base = alt.Chart(d).encode(x=alt.X("XYZ:O", title="Demand variability (X steady, Z erratic)", axis=alt.Axis(labelAngle=0)),
                               y=alt.Y("ABC:O", title="Revenue class (A highest)"))
    rect = base.mark_rect(stroke="#1c2330", strokeWidth=0).encode(
        color=alt.Color("Items:Q", scale=alt.Scale(scheme="blues"), legend=None),
        strokeWidth=alt.condition("datum.Selected", alt.value(3), alt.value(0)), tooltip=["ABC", "XYZ", "Items"])
    mid = float(d["Items"].mean())
    text = base.mark_text(fontSize=16).encode(text="Items:Q", color=alt.condition(alt.datum.Items > mid, alt.value("white"), alt.value("#1c2330")))
    return (rect + text).properties(height=210)
