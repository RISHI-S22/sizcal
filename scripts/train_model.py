from pathlib import Path
import json

import numpy as np

import xgboost as xgb

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score
)


ROOT = Path(
    __file__
).resolve().parents[1]

BRANDS_FILE = (
    ROOT / "data" / "brands.json"
)

MODEL_FILE = (
    ROOT / "models" / "sizcal_xgb.json"
)


FEATURES = [

    "shoulder_width_cm",

    "chest_circumference_cm",

    "waist_circumference_cm",

    "hip_circumference_cm",

    "inseam_cm"
]


SIZES = [

    "XS",
    "S",
    "M",
    "L",
    "XL",
    "XXL"
]


def main():

    rng = np.random.default_rng(42)

    with open(
        BRANDS_FILE,
        "r"
    ) as file:

        brands = json.load(file)

    brand_ids = list(
        brands.keys()
    )

    X = []

    y = []

    for brand_index, brand_id in enumerate(
        brand_ids
    ):

        chart = (
            brands[
                brand_id
            ]["sizes"]
        )

        for size_index, size in enumerate(
            SIZES
        ):

            base = np.array(

                [
                    chart[size][feature]

                    for feature
                    in FEATURES
                ],

                dtype=np.float32
            )

            for _ in range(150):

                noise = rng.normal(

                    0,

                    [
                        1.2,
                        2.2,
                        2.0,
                        2.2,
                        1.3
                    ]
                )

                measurement = (
                    base + noise
                )

                brand_vector = np.zeros(
                    len(brand_ids)
                )

                brand_vector[
                    brand_index
                ] = 1

                feature_vector = np.hstack(

                    [
                        measurement,
                        brand_vector
                    ]
                )

                X.append(
                    feature_vector
                )

                y.append(
                    size_index
                )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y
        )
    )

    model = xgb.XGBClassifier(

        objective="multi:softprob",

        num_class=6,

        n_estimators=160,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.9,

        colsample_bytree=0.9,

        eval_metric="mlogloss",

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    MODEL_FILE.parent.mkdir(
        exist_ok=True
    )

    model.save_model(
        str(MODEL_FILE)
    )

    print(
        "SIZCAL model created."
    )

    print(
        f"Demo accuracy: {accuracy:.4f}"
    )

    print(
        f"Saved to: {MODEL_FILE}"
    )


if __name__ == "__main__":

    main()