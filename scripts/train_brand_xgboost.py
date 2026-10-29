"""
Train brand-specific XGBoost size recommendation models.

Brands:
- Nike
- PUMA
- Levi's
- UNIQLO

The training samples are synthetic perturbations generated around
brand size-chart measurements.

This is a prototype training approach. It does NOT represent real-world
customer fit data.
"""

from pathlib import Path
import json

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRANDS_FILE = PROJECT_ROOT / "data" / "brands.json"
MODEL_DIR = PROJECT_ROOT / "models" / "brand_models"

RANDOM_SEED = 42
SAMPLES_PER_SIZE = 500

FEATURE_NAMES = [
    "shoulder_width_cm",
    "chest_circumference_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "inseam_cm",
]


def load_brands():
    """Load brand size-chart data."""
    if not BRANDS_FILE.exists():
        raise FileNotFoundError(
            f"Brand data file not found:\n{BRANDS_FILE}"
        )

    with open(BRANDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_samples_for_brand(brand_data):
    """
    Generate synthetic body measurements around each size-chart range.

    Each size gets SAMPLES_PER_SIZE examples.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    sizes = list(brand_data["sizes"].keys())

    X = []
    y = []

    for size_index, size_name in enumerate(sizes):

        size_data = brand_data["sizes"][size_name]

        # Read ranges from the brand chart.
        chest_min = float(size_data["chest"][0])
        chest_max = float(size_data["chest"][1])

        waist_min = float(size_data["waist"][0])
        waist_max = float(size_data["waist"][1])
        # Some brand charts do not provide hip measurements.
        # In that case, estimate hip from the available chest range.
        #
        # IMPORTANT:
        # This is a derived prototype feature, not an official
        # UNIQLO hip measurement.
        if "hip" in size_data:
            hip_range = size_data["hip"]
        else:
            hip_range = [
                chest_min * 0.98,
                chest_max * 0.98,
            ]

        hip_min = float(hip_range[0])
        hip_max = float(hip_range[1])

        # Some charts may not provide shoulder/inseam.
        shoulder_range = size_data.get(
            "shoulder",
            [
                0.42 * chest_min,
                0.42 * chest_max,
            ],
        )

        inseam_range = size_data.get(
            "inseam",
            [76.0, 86.0],
        )

        shoulder_min = float(shoulder_range[0])
        shoulder_max = float(shoulder_range[1])

        inseam_min = float(inseam_range[0])
        inseam_max = float(inseam_range[1])

        # Center of the size-chart range.
        chest_center = (chest_min + chest_max) / 2.0
        waist_center = (waist_min + waist_max) / 2.0
        hip_center = (hip_min + hip_max) / 2.0
        shoulder_center = (shoulder_min + shoulder_max) / 2.0
        inseam_center = (inseam_min + inseam_max) / 2.0

        # Generate realistic variation around each size.
        for _ in range(SAMPLES_PER_SIZE):

            chest = rng.normal(
                chest_center,
                max((chest_max - chest_min) / 3.0, 2.0),
            )

            waist = rng.normal(
                waist_center,
                max((waist_max - waist_min) / 3.0, 2.0),
            )

            hip = rng.normal(
                hip_center,
                max((hip_max - hip_min) / 3.0, 2.0),
            )

            shoulder = rng.normal(
                shoulder_center,
                max((shoulder_max - shoulder_min) / 3.0, 1.0),
            )

            inseam = rng.normal(
                inseam_center,
                max((inseam_max - inseam_min) / 3.0, 1.5),
            )

            # Keep values physically reasonable.
            chest = max(chest, 50.0)
            waist = max(waist, 40.0)
            hip = max(hip, 50.0)
            shoulder = max(shoulder, 25.0)
            inseam = max(inseam, 50.0)

            X.append([
                shoulder,
                chest,
                waist,
                hip,
                inseam,
            ])

            y.append(size_index)

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def train_brand_model(brand_id, brand_data):
    """Train and save one XGBoost model for one brand."""

    print()
    print("=" * 70)
    print(f"TRAINING BRAND: {brand_data['name']}")
    print("=" * 70)

    X, y = generate_samples_for_brand(brand_data)

    sizes = list(brand_data["sizes"].keys())

    print(f"Brand: {brand_data['name']}")
    print(f"Classes: {sizes}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {FEATURE_NAMES}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(sizes),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=2,
        gamma=0.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print()
    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    predictions = model.predict(X_val).astype(int)

    accuracy = accuracy_score(y_val, predictions)

    print()
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    print()
    print("Classification report:")

    print(
        classification_report(
            y_val,
            predictions,
            target_names=sizes,
            zero_division=0,
        )
    )

    # Feature importance
    importance = dict(
        zip(
            FEATURE_NAMES,
            model.feature_importances_.tolist(),
        )
    )

    print("Feature importance:")

    for feature, value in sorted(
        importance.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {feature}: {value:.4f}")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / f"{brand_id}.json"

    model.save_model(str(model_path))

    metadata = {
        "brand_id": brand_id,
        "brand_name": brand_data["name"],
        "features": FEATURE_NAMES,
        "sizes": sizes,
        "num_classes": len(sizes),
        "training_samples": int(len(X)),
        "validation_samples": int(len(X_val)),
        "validation_accuracy": float(accuracy),
        "synthetic_training": True,
        "random_seed": RANDOM_SEED,
    }

    metadata_path = MODEL_DIR / f"{brand_id}_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(f"Model saved:")
    print(f"  {model_path}")

    print(f"Metadata saved:")
    print(f"  {metadata_path}")

    return {
        "brand": brand_data["name"],
        "accuracy": accuracy,
        "samples": len(X),
        "model": str(model_path),
    }


def main():

    print("=" * 70)
    print("SIZCAL BRAND-SPECIFIC XGBOOST TRAINING")
    print("=" * 70)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Brand file:   {BRANDS_FILE}")
    print(f"Model folder: {MODEL_DIR}")

    brands = load_brands()

    print()
    print("Available brands:")

    for brand_id, brand_data in brands.items():
        print(
            f"  {brand_id}: "
            f"{brand_data['name']}"
        )

    results = []

    for brand_id, brand_data in brands.items():

        try:
            result = train_brand_model(
                brand_id,
                brand_data,
            )

            results.append(result)

        except Exception as e:

            print()
            print(
                f"ERROR training {brand_id}: "
                f"{e}"
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()

    if not results:
        print("No brand models were successfully trained.")
        return

    for result in results:

        print(
            f"{result['brand']}: "
            f"{result['accuracy'] * 100:.2f}% "
            f"validation accuracy"
        )

    print()
    print("Models created in:")

    print(MODEL_DIR)

    print()
    print("Files:")

    for path in sorted(MODEL_DIR.glob("*")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()