"""Transfers between stores and one prioritised action per store-product line."""
import numpy as np
import pandas as pd

from config import CLEAR_EXCESS_SHARE, MIN_TRANSFER, PRIORITY_ORDER
from theme import money


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


