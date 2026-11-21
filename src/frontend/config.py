"""Dashboard settings: file locations, colours, labels and planning limits."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "retail_store_inventory_processed.csv"
RAW_FILE = ROOT_DIR / "data" / "raw" / "retail_sales_forecast_dataset.csv"
MODEL_DIR = ROOT_DIR / "model"     # model/<name>/outputs/ is written by src/backend/pipeline.py
DEMAND_DIR = MODEL_DIR / "xgboost_demand" / "outputs"   # XGBoost demand model (sizes the orders)

CURRENCY = ""  # the dataset has no currency; set e.g. "RM " to prefix every value shown

REQUIRED_COLUMNS = [
    "transaction_id", "date", "store_id", "product_id", "category", "product_subcategory", "region",
    "inventory_level", "units_sold", "price", "discount",
]

INK = "#1C2B3A"
MUTED = "#5B6B7B"
BLUE = "#2B4C7E"
GRID = "#E1E7ED"
STATUS_COLORS = {"Understock risk": "#C4384A", "Healthy": "#26796B", "Overstock": "#D79A1F"}
STATUS_TEXT = {"Understock risk": "white", "Healthy": "white", "Overstock": INK}
STATUS_ORDER = list(STATUS_COLORS)

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low", "None"]
PRIORITY_COLORS = {"Critical": "#8E1F30", "High": "#C4384A", "Medium": "#D79A1F", "Low": "#8FA0B3"}
# Soft tints for table cells: (background, text)
PRIORITY_TINT = {
    "Critical": ("#F2C4CA", "#6F1423"),
    "High": ("#F7DDE0", "#8E2434"),
    "Medium": ("#F8E8C8", "#7A5410"),
    "Low": ("#E3E9EF", "#34495E"),
}
ACTION_ORDER = [
    "Order now", "Transfer in + order", "Transfer in", "Transfer out",
    "Clear excess", "Pause ordering", "Reorder soon",
]

MIN_HISTORY_RECORDS = 30  # records a subcategory needs before its average demand is trusted (used without the demand model)
MIN_TRANSFER = 10         # smallest store-to-store transfer worth suggesting (units)
CLEAR_EXCESS_SHARE = 0.6  # surplus this far above the overstock level triggers "Clear excess"
