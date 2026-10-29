from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRANDS_FILE = PROJECT_ROOT / "data" / "brands.json"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "sizcal_xgb.json"
FEATURE_INFO_PATH = MODEL_DIR / "sizcal_xgb_features.json"
METRICS_PATH = MODEL_DIR / "sizcal_xgb_metrics.json"

RANDOM_STATE = 42


# ---------------------------------------------------------
# Feature definition
# ---------------------------------------------------------

FEATURE_NAMES = [
    "shoulder_width_cm",
    "chest_circumference_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "inseam_cm",
]


SIZE_ORDER = [
    "XS",
    "S",
    "M",
    "L",
    "XL",
    "XXL",
]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_brands():

    if not BRANDS_FILE.exists():
        raise FileNotFoundError(
            f"Brand file not found:\n{BRANDS_FILE}"
        )

    with BRANDS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        brands = json.load(f)

    return brands


def normalize_size_name(size):

    return str(size).strip().upper()


def get_size_rows(brands):

    rows = []

    for brand_name, brand_data in brands.items():

        size_chart = brand_data.get(
            "size_chart",
            brand_data.get("sizes"),
        )

        if not size_chart:
            continue

        for size_name, measurements in size_chart.items():

            size_name = normalize_size_name(
                size_name
            )

            row = {
                "brand": brand_name,
                "size": size_name,
            }

            for feature in FEATURE_NAMES:

                value = measurements.get(
                    feature
                )

                if value is None:
                    value = np.nan

                row[feature] = float(value)

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No usable size-chart rows found "
            "in brands.json."
        )

    return pd.DataFrame(rows)


def interpolate_measurement(
    low,
    high,
    rng,
):

    if np.isnan(low):
        return high

    if np.isnan(high):
        return low

    return rng.uniform(
        min(low, high),
        max(low, high),
    )


def create_training_samples(
    size_rows,
    samples_per_size=300,
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    samples = []

    # Generate realistic perturbations around
    # each size-chart measurement.
    #
    # This does NOT represent measured human data.
    # It creates a controlled training set from
    # the available brand size charts.

    for _, row in size_rows.iterrows():

        brand = row["brand"]
        size = row["size"]

        if size not in SIZE_ORDER:
            continue

        center = np.array(
            [
                row[name]
                for name in FEATURE_NAMES
            ],
            dtype=np.float32,
        )

        if np.isnan(center).any():
            continue

        for _ in range(samples_per_size):

            noise = rng.normal(
                loc=0.0,
                scale=0.035,
                size=len(FEATURE_NAMES),
            )

            measurements = (
                center * (1.0 + noise)
            )

            sample = {
                "brand": brand,
                "size": size,
            }

            for i, feature in enumerate(
                FEATURE_NAMES
            ):
                sample[feature] = float(
                    measurements[i]
                )

            samples.append(sample)

    if not samples:
        raise RuntimeError(
            "No training samples were generated."
        )

    return pd.DataFrame(samples)


def encode_labels(df):

    size_to_label = {
        size: index
        for index, size
        in enumerate(SIZE_ORDER)
    }

    df = df[
        df["size"].isin(size_to_label)
    ].copy()

    df["label"] = df["size"].map(
        size_to_label
    )

    return df, size_to_label


# ---------------------------------------------------------
# Main training
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("SIZCAL - XGBOOST SIZE RECOMMENDER TRAINING")
    print("=" * 70)

    print()
    print(f"Brand file:")
    print(BRANDS_FILE)

    brands = load_brands()

    print()
    print(
        f"Brands loaded: {len(brands)}"
    )

    # -----------------------------------------------------
    # Load size charts
    # -----------------------------------------------------

    size_rows = get_size_rows(
        brands
    )

    print()
    print(
        f"Size-chart rows: {len(size_rows)}"
    )

    print()
    print("Size-chart data:")
    print(
        size_rows.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Generate training data
    # -----------------------------------------------------

    df = create_training_samples(
        size_rows,
        samples_per_size=300,
    )

    df, size_to_label = encode_labels(
        df
    )

    print()
    print(
        f"Generated training samples: "
        f"{len(df):,}"
    )

    print()
    print("Class distribution:")
    print(
        df["size"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -----------------------------------------------------
    # Features / labels
    # -----------------------------------------------------

    X = df[
        FEATURE_NAMES
    ].astype(np.float32)

    y = df["label"].astype(int)

    # -----------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print()
    print(
        f"Training samples:   {len(X_train):,}"
    )

    print(
        f"Validation samples: {len(X_val):,}"
    )

    print()
    print("Training XGBoost...")

    # -----------------------------------------------------
    # XGBoost multiclass classifier
    # -----------------------------------------------------

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(SIZE_ORDER),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],
        verbose=False,
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    predictions = model.predict(
        X_val
    ).astype(int)

    probabilities = model.predict_proba(
        X_val
    )

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Validation accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=list(
                range(len(SIZE_ORDER))
            ),
            target_names=SIZE_ORDER,
            zero_division=0,
        )
    )

    print()
    print("Confusion matrix:")

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=list(
            range(len(SIZE_ORDER))
        ),
    )

    print(
        pd.DataFrame(
            cm,
            index=SIZE_ORDER,
            columns=SIZE_ORDER,
        )
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    print()
    print("Feature importance:")

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        str(MODEL_PATH)
    )

    # -----------------------------------------------------
    # Save feature metadata
    # -----------------------------------------------------

    feature_info = {
        "feature_names": FEATURE_NAMES,
        "size_order": SIZE_ORDER,
        "size_to_label": size_to_label,
        "training_type": (
            "synthetic perturbations generated "
            "from brand size-chart measurements"
        ),
        "warning": (
            "This model is not trained on "
            "person-level ground-truth body "
            "measurements. It is a size-chart-derived "
            "baseline model."
        ),
    }

    with FEATURE_INFO_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_info,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    metrics = {
        "validation_accuracy": float(
            accuracy
        ),
        "num_training_samples": int(
            len(X_train)
        ),
        "num_validation_samples": int(
            len(X_val)
        ),
        "num_classes": len(
            SIZE_ORDER
        ),
        "classes": SIZE_ORDER,
        "feature_importance": {
            row["feature"]: float(
                row["importance"]
            )
            for _, row in importance.iterrows()
        },
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Feature metadata:")
    print(FEATURE_INFO_PATH)

    print()
    print("Metrics:")
    print(METRICS_PATH)


if __name__ == "__main__":
    main()