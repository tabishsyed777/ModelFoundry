"""Training and reproducible code generation for ModelFoundry."""

from __future__ import annotations

import io
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
    "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=120, l2_regularization=1.0, random_state=42),
}
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=180, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=120, l2_regularization=1.0, random_state=42),
}


def infer_task(target: pd.Series) -> str:
    """Suggest a task type based on target values; user can override it."""
    return "classification" if (not pd.api.types.is_numeric_dtype(target) or target.nunique() <= 12) else "regression"


def sample_dataset() -> pd.DataFrame:
    """Return a deterministic, clearly synthetic customer churn dataset for a quick demo."""
    rng = np.random.default_rng(47)
    rows = 420
    tenure = rng.integers(1, 73, rows)
    monthly_charge = np.clip(rng.normal(74, 22, rows), 18, 155).round(2)
    support_calls = rng.poisson(np.clip(2.3 - tenure / 55, 0.5, None), rows)
    late_payments = rng.binomial(4, np.clip(0.16 + monthly_charge / 500, 0.08, 0.48), rows)
    plan = rng.choice(["Basic", "Plus", "Premium"], rows, p=[0.43, 0.39, 0.18])
    plan_effect = np.select([plan == "Basic", plan == "Premium"], [0.25, -0.30], default=0.0)
    score = -2.1 + 0.035 * (monthly_charge - 70) + 0.31 * support_calls + 0.38 * late_payments - 0.033 * tenure + plan_effect
    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charge": monthly_charge,
        "support_calls_90d": support_calls,
        "late_payments_12m": late_payments,
        "plan": plan,
        "churned": np.where(churn == 1, "Yes", "No"),
    })


def build_pipeline(frame: pd.DataFrame, estimator: Any) -> Pipeline:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocess", prep), ("model", estimator)])


def _metric_values(task: str, truth: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {
            "Accuracy": float(accuracy_score(truth, prediction)),
            "Balanced accuracy": float(balanced_accuracy_score(truth, prediction)),
            "F1 (weighted)": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        }
    return {
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "R²": float(r2_score(truth, prediction)),
    }


def train_models(
    data: pd.DataFrame,
    target: str,
    task: str,
    algorithms: list[str],
    test_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit requested candidates against a shared holdout split."""
    clean = data.dropna(subset=[target]).copy()
    X = clean.drop(columns=[target])
    y = clean[target]
    if X.empty:
        raise ValueError("Keep at least one feature column besides the target.")
    if y.nunique() < 2:
        raise ValueError("The target needs at least two distinct values after empty rows are removed.")
    if task == "classification" and y.nunique() > 100:
        raise ValueError("This target has over 100 distinct classes. Choose a numeric regression target or review the target column.")
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=stratify
    )
    catalog = CLASSIFIERS if task == "classification" else REGRESSORS
    fitted: dict[str, Any] = {}
    rows = []
    for name in algorithms:
        if name not in catalog:
            continue
        pipe = build_pipeline(X_train, catalog[name])
        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test)
        rows.append({"Algorithm": name, **_metric_values(task, y_test, prediction)})
        fitted[name] = pipe
    if not rows:
        raise ValueError("Select at least one supported algorithm.")
    metrics = pd.DataFrame(rows)
    primary = "F1 (weighted)" if task == "classification" else "RMSE"
    metrics = metrics.sort_values(primary, ascending=(primary == "RMSE")).reset_index(drop=True)
    result = {
        "models": fitted,
        "X_test": X_test,
        "y_test": y_test,
        "primary_metric": primary,
        "task": task,
        "target": target,
        "feature_columns": X.columns.tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "removed_target_rows": len(data) - len(clean),
    }
    return metrics, result


def export_model(model: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.getvalue()


def generate_script(target: str, task: str, algorithms: list[str], test_fraction: float) -> str:
    """Create a self-contained script that mirrors ModelFoundry preprocessing and evaluation."""
    selected = [name for name in algorithms if name in (CLASSIFIERS if task == "classification" else REGRESSORS)]
    # JSON strings are valid Python string literals for normal CSV column names.
    target_literal = json.dumps(target, ensure_ascii=False)
    algorithms_literal = repr(selected)
    estimator_catalog = "CLASSIFIERS" if task == "classification" else "REGRESSORS"
    return f'''"""Generated by ModelFoundry. Review the data split and metrics before use."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

DATA_PATH = "your_dataset.csv"  # Change this to your CSV file
TARGET = {target_literal}
TASK = {task!r}
TEST_SIZE = {test_fraction!r}
ALGORITHMS = {algorithms_literal}

data = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])
X, y = data.drop(columns=[TARGET]), data[TARGET]
numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical = [column for column in X.columns if column not in numeric]
transformers = []
if numeric:
    transformers.append(("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), numeric))
if categorical:
    transformers.append(("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)),
    ]), categorical))
preprocess = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if TASK == "classification":
    catalog = {{
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=120, l2_regularization=1.0, random_state=42),
    }}
    stratify = y if y.value_counts().min() >= 2 else None
else:
    catalog = {{
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=180, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=120, l2_regularization=1.0, random_state=42),
    }}
    stratify = None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=stratify)
results = []
fitted = {{}}
for name in ALGORITHMS:
    pipeline = Pipeline([("preprocess", preprocess), ("model", catalog[name])])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    if TASK == "classification":
        row = {{"Algorithm": name,
               "Accuracy": accuracy_score(y_test, prediction),
               "Balanced accuracy": balanced_accuracy_score(y_test, prediction),
               "F1 (weighted)": f1_score(y_test, prediction, average="weighted", zero_division=0)}}
    else:
        row = {{"Algorithm": name,
               "MAE": mean_absolute_error(y_test, prediction),
               "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
               "R²": r2_score(y_test, prediction)}}
    results.append(row)
    fitted[name] = pipeline
table = pd.DataFrame(results)
print(table.to_string(index=False))
best = table.sort_values("F1 (weighted)" if TASK == "classification" else "RMSE",
                         ascending=TASK != "classification").iloc[0]["Algorithm"]
joblib.dump(fitted[best], "best_model.joblib", compress=3)
print(f"\\nSaved best model: {{best}} -> best_model.joblib")
'''
