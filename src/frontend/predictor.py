"""Loads the trained models and turns form inputs / test rows into predictions."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .config import FEATURES, MODELS, PREPROCESSOR_FILE, ModelSpec


class LoadedModel:
    """One trained model behind a single predict(X) call. X = encoded model inputs (as in train.csv)."""

    def __init__(self, spec: ModelSpec):
        obj = joblib.load(spec.file)
        self.name, self.kind = spec.name, spec.kind
        if spec.kind == "mlp":                       # list of networks to average
            self.members, self.columns = obj["models"], obj["features"]
        elif spec.kind == "linear":
            self.members, self.columns = [obj["model"]], obj["features"]
        else:
            names = getattr(obj, "feature_names_in_", None)
            self.members, self.columns = [obj], (list(names) if names is not None else [])
        self.estimator = obj

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns] if self.columns else X
        pred = np.mean([m.predict(X) for m in self.members], axis=0)
        return np.clip(pred, 0, None)                 # units sold cannot be negative


def available_models():
    return [m for m in MODELS if m.file.exists()]


@st.cache_resource(show_spinner="Loading model...")
def load_model(name, _mtime):
    spec = next(m for m in MODELS if m.name == name)
    return LoadedModel(spec)


def get_model(spec: ModelSpec) -> LoadedModel:
    return load_model(spec.name, spec.file.stat().st_mtime)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_FILE)


def build_inputs(**v) -> pd.DataFrame:
    """One processed-format row from form values (the same derived columns as data_transforming)."""
    price, disc = v["unit_price"], v["discount"]
    row = {
        "Promotion": int(v["promotion"]), "Holiday": int(v["holiday"]), "Month": int(v["month"]),
        "Unit_Price": price, "Discount_Percent": disc, "Marketing_Spend": v["marketing"],
        "Customer_Footfall": v["footfall"],
        "Log_Effective_Price": float(np.log(price * (1 - disc / 100))),
        "Price_Gap": price / v["competitor_price"],
        "Product_Category": v["category"], "Product_Subcategory": v["subcategory"],
        "Weather_Condition": v["weather"], "Store_Type": v["store_type"],
    }
    return pd.DataFrame([row])[FEATURES]


def predict_values(model: LoadedModel, value_sets) -> np.ndarray:
    """Predictions for several scenarios at once (each a dict of the build_inputs form values)."""
    rows = pd.concat([build_inputs(**v) for v in value_sets], ignore_index=True)
    return model.predict(encode(rows))


def encode(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Scale / one-hot rows with the preprocessor saved by the pipeline (so they match the training rows)."""
    return load_preprocessor().transform(df_processed[FEATURES])


@st.cache_data(show_spinner="Scoring the test period...")
def test_predictions(name, _mtime, _X):
    spec = next(m for m in MODELS if m.name == name)
    return get_model(spec).predict(_X)


def predict_test(spec: ModelSpec, X) -> np.ndarray:
    return test_predictions(spec.name, spec.file.stat().st_mtime, X)