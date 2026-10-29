from pathlib import Path
import json
import random

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parent.parent
BRANDS_FILE = ROOT / "data" / "brands.json"
MODELS_DIR = ROOT / "models" / "brand_models"
METRICS_FILE = ROOT / "models" / "brand_xgb_metrics.json"

FEATURES = [
    "shoulder_width_cm",
    "chest_circumference_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "inseam_cm",
]

random.seed(42)
np.random.seed(42)


def center(bounds):
    return (bounds[0] + bounds[1]) / 2.0


def width(bounds):
    return max(bounds[1] - bounds[0], 1.0)


def make_training_data(brand):
    rows = []
    labels = []

    sizes = list(brand["sizes"].items())

    for size, chart in sizes:

        chest = chart["chest"]
        waist = chart["waist"]

        hip = chart.get("hip", None)

        chest_c = center(chest)
        waist_c = center(waist)

        if hip:
            hip_c = center(hip)
        else:
            # Approximate missing hip from chest for brands whose
            # available chart only exposes chest and waist.
            hip_c = chest_c

        # Shoulder/inseam are not part of the brand chart.
        # They are generated as auxiliary features, not ground truth.
        shoulder_c = chest_c * 0.43
        inseam_c = 0.45 * 170.0

        for _ in range(700):

            chest_value = np.random.normal(
                chest_c,
                width(chest) * 0.23
            )

            waist_value = np.random.normal(
                waist_c,
                width(waist) * 0.23
            )

            hip_value = np.random.normal(
                hip_c,
                width(hip) * 0.23 if hip else width(chest) * 0.23
            )

            shoulder_value = np.random.normal(
                shoulder_c,
                1.8
            )

            inseam_value = np.random.normal(
                inseam_c,
                3.0
            )

            rows.append([
                shoulder_value,
                chest_value,
                waist_value,
                hip_value,
                inseam_value,
            ])

            labels.append(size)

    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


def train_brand(brand_id, brand):
    X, y_text = make_training_data(brand)

    class_names = list(brand["sizes"].keys())
    class_to_id = {
        name: index
        for index, name in enumerate(class_names)
    }

    y = np.array([
        class_to_id[value]
        for value in y_text
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    prediction = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        prediction
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODELS_DIR / f"{brand_id}.json"
    model.save_model(str(model_path))

    report = classification_report(
        y_val,
        prediction,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "brand": brand["name"],
        "brand_id": brand_id,
        "samples": int(len(X)),
        "validation_accuracy": round(
            float(accuracy) * 100,
            2
        ),
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
        "classes": class_names,
        "classification_report": report,
    }


def main():
    brands = json.loads(
        BRANDS_FILE.read_text(
            encoding="utf-8"
        )
    )

    metrics = {}

    print("=" * 70)
    print("SIZCAL MULTI-BRAND XGBOOST TRAINING")
    print("=" * 70)

    for brand_id, brand in brands.items():

        print()
        print(f"Training: {brand['name']}")

        result = train_brand(
            brand_id,
            brand
        )

        metrics[brand_id] = result

        print(
            f"Validation accuracy: "
            f"{result['validation_accuracy']:.2f}%"
        )

    (ROOT / "models").mkdir(
        exist_ok=True
    )

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("MULTI-BRAND TRAINING COMPLETE")
    print("=" * 70)
    print(f"Saved metrics: {METRICS_FILE}")


if __name__ == "__main__":
    main()
