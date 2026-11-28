"""Tab 3, Inventory recommender: reorder point, safety stock and stockout risk for each record, ABC-XYZ segments, priority list."""
import numpy as np
import pandas as pd
import streamlit as st

from . import charts, data, predictor, recommender as rec
from .config import DATE_COL, STATUS_COLORS
from .ui import STRETCH


def build_table(meta, X, spec, rules, lead, sigma, sigma_all, period):
    pred = predictor.predict_test(spec, X)
    table = pd.concat([meta.reset_index(drop=True), pd.Series(pred, name="Predicted_Units")], axis=1)
    table = table[(table[DATE_COL].dt.date >= period[0]) & (table[DATE_COL].dt.date <= period[1])].reset_index(drop=True)
    if table.empty:
        return table
    sig = table["Product_Category"].map(sigma).fillna(sigma_all).to_numpy(float)
    table = pd.concat([table, rec.plan(table["Predicted_Units"], table["Inventory_Level"], lead.for_rows(table), sig, rules)], axis=1)
    seg = data.abc_xyz_table()["Segment"]
    table["Segment"] = table["Product_ID"].map(seg)
    return table


def item_card(table, rules):
    with st.container(border=True):
        a, b = st.columns(2)
        store = a.selectbox("Store", sorted(table["Store_ID"].unique()))
        products = sorted(table.loc[table["Store_ID"] == store, "Product_ID"].unique())
        product = b.selectbox("Product", products)
        row = table[(table["Store_ID"] == store) & (table["Product_ID"] == product)].sort_values(DATE_COL).iloc[-1]
        k = st.columns(4)
        k[0].metric("Stock on hand", f"{int(row['Inventory_Level']):,}")
        k[1].metric("Reorder point", f"{row['Reorder_Point']:,.0f}")
        k[2].metric("Safety stock", f"{row['Safety_Stock']:,.0f}")
        k[3].metric("Stockout risk", f"{row['Stockout_Risk']:.0%}")
        st.caption(f"{row['Product_Category']} / {row['Product_Subcategory']}, record of {row[DATE_COL].date()}: predicted {row['Predicted_Units']:.1f} units a day, "
                   f"lead time {row['Lead_Days']:.0f} days + {rules.review_days} review days, service level {rules.service_level:.0%}.")
    return row


def render(meta, X, spec, rules, lead, sigma, sigma_all, period):
    table = build_table(meta, X, spec, rules, lead, sigma, sigma_all, period)
    if table.empty:
        st.info("No records in this date window.")
        return
    row = item_card(table, rules)

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Stock vs demand over the lead time + review period**")
        st.altair_chart(charts.stock_vs_plan(int(row["Inventory_Level"]), row), **STRETCH)
        {"Understocking": st.error, "Overstocking": st.warning, "Sufficient stock": st.success}[row["Status"]](f"**{row['Status']}.** {row['Action']}.")
    with right, st.container(border=True):
        st.markdown("**ABC-XYZ segments (products)**")
        seg = data.abc_xyz_table()
        counts = seg.groupby(["ABC", "XYZ"]).size().rename("Items").reset_index()
        full = pd.MultiIndex.from_product([["A", "B", "C"], ["X", "Y", "Z"]], names=["ABC", "XYZ"]).to_frame(index=False)
        counts = full.merge(counts, how="left").fillna({"Items": 0})
        st.altair_chart(charts.abc_xyz_matrix(counts, row["Segment"]), **STRETCH)
        st.caption(f"This product: **{row['Segment']}**. {rec.SEGMENT_POLICY.get(row['Segment'], '')}")

    with st.container(border=True):
        st.markdown("##### Priority list")
        f1, f2, f3, f4 = st.columns(4)
        stores = f1.multiselect("Store", sorted(table["Store_ID"].unique()), placeholder="All stores")
        cats = f2.multiselect("Category", sorted(table["Product_Category"].dropna().unique()), placeholder="All categories")
        statuses = f3.multiselect("Status", list(STATUS_COLORS), placeholder="All statuses")
        segs = f4.multiselect("Segment", sorted(table["Segment"].dropna().unique()), placeholder="All segments")
        view = table
        for col, chosen in (("Store_ID", stores), ("Product_Category", cats), ("Status", statuses), ("Segment", segs)):
            if chosen:
                view = view[view[col].isin(chosen)]
        if view.empty:
            st.info("No records match these filters.")
            return
        n = view["Status"].value_counts()
        st.caption(f"{len(view):,} records: " + ", ".join(f"{rec.label(s)} {int(n.get(s, 0)):,}" for s in STATUS_COLORS)
                   + ". Highest priority first: higher stockout risk (understocking) and larger excess (overstocking) rank first; high-demand items get a bonus.")
        show = view.sort_values(["Priority", "Stockout_Risk"], ascending=False).head(500).copy()
        show["Status"] = show["Status"].map(rec.label)
        show[DATE_COL] = show[DATE_COL].dt.date
        show["Stockout_Risk"] = show["Stockout_Risk"] * 100
        cols = [DATE_COL, "Store_ID", "Product_ID", "Product_Category", "Segment", "Inventory_Level", "Predicted_Units", "Reorder_Point",
                "Stockout_Risk", "Status", "Priority", "Action"]
        st.dataframe(show[cols], hide_index=True, height=420, **STRETCH, column_config={
            "Predicted_Units": st.column_config.NumberColumn("Predicted units", format="%.1f"),
            "Inventory_Level": st.column_config.NumberColumn("Stock"),
            "Reorder_Point": st.column_config.NumberColumn("Reorder point", format="%.0f"),
            "Stockout_Risk": st.column_config.NumberColumn("Stockout risk", format="%.0f%%"),
            "Priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.0f"),
            "Product_Category": "Category", "Store_ID": "Store", "Product_ID": "Product"})
        st.download_button("Download this view (CSV)", view.sort_values("Priority", ascending=False).to_csv(index=False),
                           "inventory_recommendations.csv", "text/csv")
        st.caption(f"Showing the top 500 of {len(view):,} records. Predictions: {spec.name}, on the held-out test period.")
