from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from services.eligibility.features import extract_features, feature_order

# Where to store model inside the API container (mounted volume recommended)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "eligibility_logreg.pkl"
FEATS_PATH = MODEL_DIR / "feature_names.json"


def _vectorize(feats: dict[str, float]) -> np.ndarray:
    order = feature_order()
    return np.array([[feats.get(k, 0.0) for k in order]], dtype=float)


def train_from_applications(app_ids: list[str]) -> dict:
    """
    Minimal training loop.
    - Uses current applications as 'dataset'.
    - Labels are synthetic here (demo): 1 if income_per_capita >= threshold & docs present, else 0.
      In real projects, replace with historical labels.
    """
    X, y = [], []
    order = feature_order()
    for app_id in app_ids:
        feats = extract_features(app_id)
        v = [feats[k] for k in order]
        # synthetic label: pass if income_per_capita >= 1000 and no missing docs
        label = (
            1
            if (feats["income_per_capita"] >= 1000 and feats["missing_required_count"] == 0)
            else 0
        )
        X.append(v)
        y.append(label)

    if not X:
        raise ValueError("no training data found")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipe.fit(np.array(X), np.array(y))

    # persist model + feature order
    import joblib

    joblib.dump(pipe, MODEL_PATH)
    FEATS_PATH.write_text(json.dumps(order))

    return {"trained_on": len(X), "pos_rate": float(np.mean(y)), "model_path": str(MODEL_PATH)}


def load_model() -> Pipeline:
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found at {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)
    check_is_fitted(pipe)
    return pipe


def predict_with_explanations(app_id: str) -> dict:
    """
    Returns:
      - prob_eligible
      - contributions per feature = coef * standardized_value (transparent)
    """
    feats = extract_features(app_id)
    x = _vectorize(feats)

    pipe = load_model()
    # predict proba
    proba = float(pipe.predict_proba(x)[0, 1])

    # contributions: scaler + coefficients
    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    # standardize and multiply by coefficients
    z = (x - scaler.mean_) / scaler.scale_
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    coef = clf.coef_[0]
    contribs = (z[0] * coef).tolist()

    order = feature_order()
    explanation = {order[i]: float(contribs[i]) for i in range(len(order))}

    return {
        "prob_eligible": proba,
        "features": feats,
        "contributions": explanation,  # signed impact towards eligibility
    }
