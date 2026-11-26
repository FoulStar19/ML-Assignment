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

INK = "#1E2A44"
MUTED = "#66718C"
BLUE = "#3B5BDB"
GRID = "#E4E9F3"
BG = "#F4F6FB"
VIOLET = "#7C5CE0"
SKY = "#2BB5D8"
PALETTE = [BLUE, "#22A27B", "#F5A524", VIOLET, "#E5484D", SKY]   # categorical charts

STATUS_COLORS = {"Understock risk": "#E5484D", "Healthy": "#22A27B", "Overstock": "#F5A524"}
STATUS_TEXT = {"Understock risk": "white", "Healthy": "white", "Overstock": INK}
STATUS_ORDER = list(STATUS_COLORS)
# Soft tints for chips: (background, text)
STATUS_TINT = {"Understock risk": ("#FDE3E4", "#A3262C"), "Healthy": ("#D6F2E8", "#126B4E"), "Overstock": ("#FEEBC8", "#8A5A06")}

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low", "None"]
PRIORITY_COLORS = {"Critical": "#B4232F", "High": "#E5484D", "Medium": "#F5A524", "Low": "#94A3C0"}
PRIORITY_TINT = {
    "Critical": ("#FAD4D7", "#8A1A25"),
    "High": ("#FDE3E4", "#A3262C"),
    "Medium": ("#FEEBC8", "#8A5A06"),
    "Low": ("#E6EBF5", "#40507A"),
}
ACTION_ORDER = [
    "Order now", "Transfer in + order", "Transfer in", "Transfer out",
    "Clear excess", "Pause ordering", "Reorder soon",
]
ACTION_ICONS = {
    "Order now": "🛒", "Transfer in + order": "🔁", "Transfer in": "📥", "Transfer out": "📤",
    "Clear excess": "🏷️", "Pause ordering": "⏸️", "Reorder soon": "⏰", "No action": "✅",
}

MIN_HISTORY_RECORDS = 30  # records a subcategory needs before its average demand is trusted (used without the demand model)
MIN_TRANSFER = 10         # smallest store-to-store transfer worth suggesting (units)
CLEAR_EXCESS_SHARE = 0.6  # surplus this far above the overstock level triggers "Clear excess"
