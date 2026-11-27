from model.logistic_regression.logistic_regression_model import train_logistic_regression
from model.mlp.mlp_model import train_mlp
from model.random_forest.random_forest_model import train_random_forest
from model.xgboost.xgboost_model import train_xgboost
from model.xgboost_recommender.xgboost_recommender_model import train_xgboost_recommender

LINE = "─" * 45

# ─────────────────────────────────────────────
# 5. MODEL TRAINING, TUNING AND EVALUATION
# ─────────────────────────────────────────────
def train_models():
    # 1. Logistic Regression (baseline)
    print(LINE)
    print("1. Logistic Regression (baseline)")
    train_logistic_regression()
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

    # 5. XGBoost for recommender system (predicting Units_Sold)
    print(LINE)
    print("5. XGBoost for recommender system (predicting Units_Sold)")
    train_xgboost_recommender()
    print(LINE + "\n")

    return 