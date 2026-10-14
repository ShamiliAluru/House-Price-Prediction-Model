"""Validate the supplied Hyderabad dataset, compare models, and save the selected model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Hyderabad.csv"
MODEL_PATH = ROOT / "house_price_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

AREA = "Area"
BEDROOMS = "No. of Bedrooms"
LOCATION = "Location"
RESALE = "Resale"
PRICE = "Price"
AMENITIES = ["Gymnasium", "SwimmingPool", "CarParking", "PowerBackup", "24X7Security", "LiftAvailable"]
NUMERIC = [AREA, BEDROOMS]
CATEGORICAL = [LOCATION, RESALE, *AMENITIES]
FEATURES = [*NUMERIC, *CATEGORICAL]
MODEL_NAMES = {"Ridge": "Regularized linear regression", "ExtraTrees": "Extra Trees ensemble"}


def load_and_validate(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Validate source rows, remove only exact duplicates, and normalize source codes."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Keep Hyderabad.csv beside this script.")
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV: {exc}") from exc
    if raw.empty:
        raise ValueError("The CSV contains headers but no property rows.")
    if raw.columns.duplicated().any():
        raise ValueError("The CSV has duplicate column names. Please check the source file.")
    required = [PRICE, AREA, LOCATION, BEDROOMS, RESALE, *AMENITIES]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError("This file is missing columns required by this project: " + ", ".join(missing))

    original_rows = len(raw)
    duplicate_mask = raw.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    unique_raw = raw.loc[~duplicate_mask]
    data = unique_raw[required].copy()
    for column in [PRICE, AREA, BEDROOMS, RESALE, *AMENITIES]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[LOCATION] = data[LOCATION].astype("string").str.strip()
    amenity_codes_invalid = (unique_raw[AMENITIES].notna() & ~data[AMENITIES].isin([0, 1, 9])).any(axis=1)
    invalid = (
        data[[PRICE, AREA, BEDROOMS, RESALE]].isna().any(axis=1)
        | data[LOCATION].isna()
        | data[LOCATION].eq("")
        | amenity_codes_invalid
        | ~np.isfinite(data[[PRICE, AREA, BEDROOMS, RESALE]].to_numpy(dtype=float)).all(axis=1)
        | data[[PRICE, AREA, BEDROOMS]].le(0).any(axis=1)
        | data[BEDROOMS].mod(1).ne(0)
        | ~data[RESALE].isin([0, 1])
    )
    invalid_count = int(invalid.sum())
    data = data.loc[~invalid].copy()
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} valid unique rows remain. At least 100 are needed for the evaluation used here.")

    # In this source family, 0/1 represent No/Yes and 9 means no amenity detail was supplied.
    # Keep that as its own category; never reinterpret it as No.
    for column in AMENITIES:
        data[column] = data[column].map({0: "No", 1: "Yes", 9: "Not specified"})
        data[column] = data[column].fillna("Not specified").astype(str)
    data[RESALE] = data[RESALE].map({0: "New", 1: "Resale"})
    data[LOCATION] = data[LOCATION].astype(str)
    data[PRICE] = data[PRICE].astype(float)
    data[AREA] = data[AREA].astype(float)
    data[BEDROOMS] = data[BEDROOMS].astype(int)

    audit = {
        "source_rows": int(original_rows),
        "exact_duplicate_rows_removed": duplicate_rows,
        "invalid_rows_removed": invalid_count,
        "training_rows": int(len(data)),
        "locations": int(data[LOCATION].nunique()),
        "amenity_code_note": "0=No, 1=Yes, 9=Not specified; 9 was preserved as its own category.",
    }
    return data.reset_index(drop=True), audit


def make_pipeline(kind: str) -> Pipeline:
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
    ])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    if kind == "Ridge":
        estimator = Ridge(alpha=15.0)
    elif kind == "ExtraTrees":
        estimator = ExtraTreesRegressor(
            n_estimators=240, min_samples_leaf=3, max_features=0.85,
            random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {kind}")
    regressor = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("prepare", prep), ("predictor", regressor)])


def evaluate_and_train() -> dict:
    data, audit = load_and_validate()
    X = data[FEATURES]
    y = data[PRICE]
    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline_predictions = np.zeros(len(data), dtype=float)
    nested_predictions = np.zeros(len(data), dtype=float)
    selected_by_fold: list[str] = []

    for train_idx, test_idx in outer.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        inner = KFold(n_splits=3, shuffle=True, random_state=43)
        fold_candidates: list[tuple[float, str, Pipeline]] = []
        for name in MODEL_NAMES:
            candidate = make_pipeline(name)
            scores = cross_val_score(
                candidate, X_train, y_train, cv=inner,
                scoring="neg_mean_absolute_error", n_jobs=1,
            )
            cv_mae = -scores
            fold_candidates.append((float(np.mean(cv_mae)), name, candidate))
        _, chosen_name, chosen = min(fold_candidates, key=lambda item: item[0])
        chosen.fit(X_train, y_train)
        nested_predictions[test_idx] = chosen.predict(X_test)
        baseline_predictions[test_idx] = float(y_train.median())
        selected_by_fold.append(chosen_name)

    # Select the final deployable model from all training rows using the same inner-CV rule.
    full_cv = KFold(n_splits=5, shuffle=True, random_state=44)
    full_candidates: list[tuple[float, str, Pipeline]] = []
    final_cv_mae: dict[str, float] = {}
    for name in MODEL_NAMES:
        candidate = make_pipeline(name)
        score = -cross_val_score(candidate, X, y, cv=full_cv, scoring="neg_mean_absolute_error", n_jobs=1).mean()
        final_cv_mae[name] = float(score)
        full_candidates.append((float(score), name, candidate))
    _, selected_name, final_model = min(full_candidates, key=lambda item: item[0])
    final_model.fit(X, y)

    metrics = {
        **audit,
        "evaluation": "Nested 5-fold cross-validation; each outer training fold selected between two models using 3-fold CV.",
        "selection_metric": "Mean absolute error (MAE), measured in the dataset's price values.",
        "nested_cv_mae": round(float(mean_absolute_error(y, nested_predictions)), 2),
        "nested_cv_rmse": round(float(np.sqrt(mean_squared_error(y, nested_predictions))), 2),
        "nested_cv_r2": round(float(r2_score(y, nested_predictions)), 4),
        "median_baseline_mae": round(float(mean_absolute_error(y, baseline_predictions)), 2),
        "model_selection_cv_mae": {key: round(value, 2) for key, value in final_cv_mae.items()},
        "selected_model": selected_name,
        "selected_model_label": MODEL_NAMES[selected_name],
        "outer_fold_selected_models": selected_by_fold,
        "price_display": "₹ inferred from the Hyderabad, India dataset context; the raw CSV does not declare a currency code.",
        "limitations": [
            "Source rows have no collection date, so this is not a current-market guarantee.",
            "The data has no bathroom count or PIN code; neither is part of the model.",
            "Exact duplicate rows were removed before training.",
            "CV measures performance on held-out rows drawn from the same dataset; it is not a confidence interval or guarantee for a particular home.",
        ],
    }
    locations = data.groupby(LOCATION).size().sort_values(ascending=False)
    artifact = {
        "model": final_model,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "amenities": AMENITIES,
        "dataset_rows": int(len(data)),
        "location_counts": locations.to_dict(),
        "supported_area_range": [float(data[AREA].quantile(0.01)), float(data[AREA].quantile(0.99))],
        "supported_bedroom_values": sorted(int(x) for x in data[BEDROOMS].unique()),
        "price_range": [float(data[PRICE].min()), float(data[PRICE].max())],
        "median_price": float(data[PRICE].median()),
        "data": data,
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = evaluate_and_train()
    print(json.dumps(result, indent=2))
    print(f"\nSaved model to: {MODEL_PATH}")
