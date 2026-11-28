from model.linear_regression.linear_regression_model import main as train_linear_regression
from model.mlp.mlp_model import main as train_mlp
from model.xgboost.xgboost_model import main as train_xgboost
from model.random_forest.random_forest_model import main as train_random_forest

LINE = "─" * 45

# ─────────────────────────────────────────────
# 5. MODEL TRAINING, TUNING AND EVALUATION
# ─────────────────────────────────────────────
def train_models():
    # 1. Linear Regression (baseline)
    print(LINE)
    print("1. Linear Regression (baseline)")
    train_linear_regression()
    print(LINE + "\n")


    # 2. Random Forest
    print(LINE)
    print("2. Random Forest")
    train_random_forest()
    print(LINE + "\n")

    # 3. XGBoost
    print(LINE)
    print("3. XGBoost")
    train_xgboost()
    print(LINE + "\n")

    # 4. MLP
    print(LINE)
    print("4. MLP")
    train_mlp()
    print(LINE + "\n")

    return