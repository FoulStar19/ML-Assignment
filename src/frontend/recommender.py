"""Transfers between stores and one prioritised action per record."""
import numpy as np
import pandas as pd

from config import CLEAR_EXCESS_SHARE, MIN_TRANSFER, PRIORITY_ORDER


def plan_transfers(lines: pd.DataFrame) -> pd.DataFrame:
    """Match surplus stock to shortfalls within the same product subcategory across stores.

    Product IDs are only labels in this data (one ID is sold under many categories), so the product subcategory
    is the closest thing to "the same product". Surplus = units above a record's target level. Shortfall = units
    needed to get a record back to target. Largest shortfalls are served first.
    """
    cols = ["product_subcategory", "from_store", "from_product", "to_store", "to_product", "qty", "value"]
    rows = []
    for sub, g in lines.groupby("product_subcategory", observed=True):
        need = g[g["order_qty"] > 0].sort_values("order_qty", ascending=False)
        surplus = g[g["excess_units"] > 0]
        if need.empty or surplus.empty:
            continue
        avail = {(r.store_id, r.product_id): int(r.excess_units) for r in surplus.itertuples()}
        for n in need.itertuples():
            remaining = int(n.order_qty)
            for src in sorted(avail, key=avail.get, reverse=True):
                if src[0] == n.store_id:                       # a store cannot send stock to itself
                    continue
                qty = min(remaining, avail[src])
                if qty < MIN_TRANSFER:
                    continue
                rows.append({"product_subcategory": sub, "from_store": src[0], "from_product": src[1],
                             "to_store": n.store_id, "to_product": n.product_id,
                             "qty": qty, "value": qty * n.price})
                avail[src] -= qty
                remaining -= qty
                if remaining <= 0:
                    break
    return pd.DataFrame(rows, columns=cols)


def recommend(lines: pd.DataFrame, transfers: pd.DataFrame, lead: int, over_days: float, watch_days: float) -> pd.DataFrame:
    """One prioritised action per record, with a plain-language reason."""
    d = lines.copy().reset_index(drop=True)
    d["transfer_in"], d["transfer_out"] = 0, 0
    d["from_stores"], d["to_stores"] = "", ""
    if not transfers.empty:
        idx = pd.MultiIndex.from_frame(d[["store_id", "product_id"]])
        t = transfers.assign(from_label=transfers["from_store"] + " · " + transfers["from_product"],
                             to_label=transfers["to_store"] + " · " + transfers["to_product"])
        tin = t.groupby(["to_store", "to_product"]).agg(qty=("qty", "sum"), who=("from_label", lambda s: ", ".join(s)))
        tout = t.groupby(["from_store", "from_product"]).agg(qty=("qty", "sum"), who=("to_label", lambda s: ", ".join(s)))
        tin.index.names = tout.index.names = ["store_id", "product_id"]
        d["transfer_in"] = tin["qty"].reindex(idx).fillna(0).astype(int).to_numpy()
        d["from_stores"] = tin["who"].reindex(idx).fillna("").to_numpy()
        d["transfer_out"] = tout["qty"].reindex(idx).fillna(0).astype(int).to_numpy()
        d["to_stores"] = tout["who"].reindex(idx).fillna("").to_numpy()

    d["net_order"] = (d["order_qty"] - d["transfer_in"]).clip(lower=0)
    d["net_order_value"] = d["net_order"] * d["price"]
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
    d["risk"] = np.maximum(d["p_under"], d["p_over"]) if "p_under" in d else 0.0   # the model's confidence in a problem
    d["rank"] = d["priority"].map({p: i for i, p in enumerate(PRIORITY_ORDER)})
    return d.sort_values(["rank", "risk"], ascending=[True, False]).reset_index(drop=True)


def _reason(r: pd.Series, lead: int, over_days: float) -> str:
    """A short fragment per recommendation."""
    a = r["action"]
    cover = f"{r['days_of_cover']:.1f} d cover"
    risk = f"{r['p_under']:.0%} risk" if "p_under" in r.index else ""
    if a == "Order now" and r["priority"] == "Critical":
        return f"{cover}, under the {lead} d lead time"
    if a == "Order now":
        return f"{cover} · {risk}"
    if a == "Transfer in":
        return f"Take {r['transfer_in']:,} from {r['from_stores']}"
    if a == "Transfer in + order":
        return f"Take {r['transfer_in']:,} from {r['from_stores']}, order {r['net_order']:,.0f}"
    if a == "Transfer out":
        return f"Send {r['transfer_out']:,} to {r['to_stores']}"
    if a == "Clear excess":
        return f"{r['excess_units']:,.0f} over target ({r['excess_share']:.0%})"
    if a == "Pause ordering":
        return cover
    if a == "Reorder soon":
        return f"Reorder point in {r['days_to_reorder']:.1f} d"
    return "Healthy"
