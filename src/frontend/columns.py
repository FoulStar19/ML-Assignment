"""Column dictionary for the data explorer: what each column means and what role it plays in the project."""
from .config import DATE_COL, FEATURES, STOCK_COLS, TARGET

DESCRIPTIONS = {
    "Transaction_ID": "Unique ID of the sales record",
    "Date": "Date of the record",
    "Year": "Year taken from Date", "Month": "Month number taken from Date", "Week": "Week number taken from Date",
    "Day": "Day of the month", "Day_Of_Week": "Weekday name", "Quarter": "Quarter of the year",
    "Store_ID": "Store identifier", "Store_Type": "Kind of store (Supermarket, Hypermarket, Convenience Store, Online Store)",
    "Region": "Region of the store", "City": "City of the store",
    "Product_ID": "Product identifier", "Product_Category": "Broad product group", "Product_Subcategory": "Product group within the category",
    "Brand": "Brand of the product",
    "Units_Sold": "Units sold in the record (what the models predict)",
    "Unit_Price": "Price per unit before discount", "Discount_Percent": "Discount applied to the price, in %",
    "Promotion": "Whether a promotion was running (Yes/No)", "Holiday": "Whether the day was a holiday (Yes/No)",
    "Season": "Season of the year",
    "Inventory_Level": "Units in stock (used by the recommender)",
    "Competitor_Price": "Price of the competing product (used to build Price_Gap)",
    "Customer_Footfall": "Number of customers visiting the store", "Marketing_Spend": "Marketing spend for the record",
    "Weather_Condition": "Weather on the day", "Economic_Index": "Economic indicator for the period",
    "Online_Orders": "Number of online orders", "Returns": "Number of returned units",
    "Customer_Satisfaction": "Customer rating (1 to 5)", "Shipping_Time": "Shipping time in days (used as the lead time)",
    "Supplier_Rating": "Supplier rating (1 to 5)",
    "Sales_Amount": "Revenue = Units_Sold x Unit_Price x (1 - Discount/100)",
    "Demand_Level": "Low / Medium / High / Very High, set by bands of Sales_Amount",
    "Log_Effective_Price": "Derived: log of the price after discount",
    "Price_Gap": "Derived: Unit_Price / Competitor_Price (above 1 = dearer than the competitor)",
}


def role(col):
    if col == TARGET:
        return "Target"
    if col in FEATURES:
        return "Model feature"
    if col == "Competitor_Price":
        return "Builds Price_Gap"
    if col == DATE_COL:
        return "Builds Month and sets the time split"
    if col in STOCK_COLS:
        return "Stock info for the recommender"
    if col == "Shipping_Time":
        return "Lead time for the recommender"
    if col in ("Sales_Amount", "Demand_Level"):
        return "Not a feature (would leak the target)"
    return "Not used"


def dictionary(df):
    """One row per column of df: type, role, description."""
    return [{"Column": c, "Type": "number" if str(df[c].dtype)[:3] in ("int", "flo") else ("date" if "datetime" in str(df[c].dtype) else "text"),
             "Role": role(c), "Meaning": DESCRIPTIONS.get(c, "")} for c in df.columns]
