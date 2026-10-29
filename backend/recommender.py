from pathlib import Path
import json
import math
import numpy as np
import xgboost as xgb


FEATURES = [
    "shoulder_width_cm",
    "chest_circumference_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "inseam_cm",
]


class SIZCALRecommender:
    """
    Brand-aware size recommendation.

    Each brand has its own size chart. XGBoost models are trained
    separately for each brand from chart-derived training samples.

    The chart score remains the safety/interpretability layer:
    it prevents one global model from forcing every user into XS.
    """

    def __init__(self, project_root=None):
        self.project_root = (
            Path(project_root)
            if project_root
            else Path(__file__).resolve().parent.parent
        )

        self.brands_path = self.project_root / "data" / "brands.json"
        self.models_dir = self.project_root / "models" / "brand_models"

        self.brands = json.loads(
            self.brands_path.read_text(encoding="utf-8")
        )

        self.models = {}
        self.model_features = FEATURES

        self._load_models()

    def _load_models(self):
        for brand_id in self.brands:
            path = self.models_dir / f"{brand_id}.json"
            if path.exists():
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(str(path))
                    self.models[brand_id] = model
                except Exception as exc:
                    print(f"Could not load {brand_id} model: {exc}")

    def get_brands(self):
        return [
            {
                "id": brand_id,
                "name": info["name"],
                "category": info.get("category", ""),
            }
            for brand_id, info in self.brands.items()
        ]

    def get_sizes(self):
        sizes = []
        seen = set()

        for info in self.brands.values():
            for size in info["sizes"]:
                if size not in seen:
                    seen.add(size)
                    sizes.append(size)

        return sizes

    @staticmethod
    def _distance_to_range(value, lower, upper):
        if lower <= value <= upper:
            return 0.0
        if value < lower:
            return lower - value
        return value - upper

    @staticmethod
    def _range_center(bounds):
        return (bounds[0] + bounds[1]) / 2.0

    def _chart_score(self, measurements, size_data):
        """
        Score a size against a brand chart.
        1.0 = comfortably inside all available ranges.
        Lower values = farther from the brand's size range.
        """

        distances = []
        inside = 0

        fields = [
            ("chest_circumference_cm", "chest"),
            ("waist_circumference_cm", "waist"),
            ("hip_circumference_cm", "hip"),
        ]

        for measurement_key, chart_key in fields:
            value = measurements.get(measurement_key)

            if value is None or chart_key not in size_data:
                continue

            value = float(value)
            lower, upper = size_data[chart_key]

            distance = self._distance_to_range(
                value, lower, upper
            )

            width = max(upper - lower, 1.0)
            distances.append(distance / width)

            if distance == 0:
                inside += 1

        if not distances:
            return 0.0, 0, 0.0

        average_distance = sum(distances) / len(distances)

        score = max(
            0.0,
            1.0 - average_distance
        )

        coverage = inside / len(distances)

        return score, inside, coverage

    def _chart_recommendation(self, measurements, brand_id):
        brand = self.brands[brand_id]
        candidates = []

        for size, size_data in brand["sizes"].items():
            score, inside, coverage = self._chart_score(
                measurements,
                size_data
            )

            candidates.append({
                "size": size,
                "score": score,
                "inside": inside,
                "coverage": coverage,
            })

        candidates.sort(
            key=lambda item: (
                item["inside"],
                item["score"],
            ),
            reverse=True
        )

        best = candidates[0]

        # If a user is between sizes, prefer the larger size when
        # one or more important measurements are outside the smaller size.
        confidence = round(
            min(
                99.0,
                max(
                    45.0,
                    best["score"] * 100.0
                )
            ),
            2
        )

        return {
            "size": best["size"],
            "chart_score": round(best["score"], 4),
            "confidence_percent": confidence,
            "measurements_inside_range": best["inside"],
            "available_measurements": int(best["coverage"] * 3),
        }

    def _xgb_recommendation(self, measurements, brand_id):
        model = self.models.get(brand_id)

        if model is None:
            return None

        row = np.array(
            [[
                float(measurements[f])
                for f in FEATURES
            ]],
            dtype=np.float32
        )

        try:
            probabilities = model.predict_proba(row)[0]

            # XGBoost stores the encoded numeric class labels.
            # Convert them back to the actual brand size names.
            size_names = list(
                self.brands[brand_id]["sizes"].keys()
            )

            classes = [
                size_names[int(x)]
                for x in model.classes_
            ]

            ranking = sorted(
                zip(classes, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            return {
                "size": ranking[0][0],
                "probabilities": [
                    {
                        "size": size,
                        "confidence_percent": round(
                            float(prob) * 100.0,
                            2
                        ),
                    }
                    for size, prob in ranking
                ],
            }

        except Exception as exc:
            print(
                f"XGBoost prediction failed for {brand_id}: {exc}"
            )
            return None

    def recommend_brand(self, measurements, brand_id):
        if brand_id not in self.brands:
            raise ValueError(
                f"Unknown brand: {brand_id}"
            )

        for feature in FEATURES:
            if feature not in measurements:
                raise ValueError(
                    f"Missing measurement: {feature}"
                )

            value = float(measurements[feature])

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid measurement: {feature}"
                )

        chart = self._chart_recommendation(
            measurements,
            brand_id
        )

        xgb_result = self._xgb_recommendation(
            measurements,
            brand_id
        )

        # Chart result is the primary decision.
        # XGBoost is reported as supporting model output.
        final_size = chart["size"]

        if xgb_result:
            model_size = xgb_result["size"]

            if model_size == final_size:
                model_agreement = True
            else:
                model_agreement = False
        else:
            model_size = None
            model_agreement = None

        return {
            "brand_id": brand_id,
            "brand_name": self.brands[brand_id]["name"],
            "recommended_size": final_size,
            "confidence_percent": chart["confidence_percent"],
            "chart_score": chart["chart_score"],
            "model_size": model_size,
            "model_agreement": model_agreement,
            "xgboost_available": xgb_result is not None,
            "probabilities": (
                xgb_result["probabilities"]
                if xgb_result
                else []
            ),
        }

    def recommend_all_brands(self, measurements):
        results = []

        for brand_id in self.brands:
            try:
                result = self.recommend_brand(
                    measurements,
                    brand_id
                )

                results.append(result)

            except Exception as exc:
                print(
                    f"Brand recommendation failed "
                    f"for {brand_id}: {exc}"
                )

        results.sort(
            key=lambda item: (
                item["chart_score"],
                item["confidence_percent"],
            ),
            reverse=True
        )

        for index, result in enumerate(results):
            result["rank"] = index + 1

        return results

    def recommend(self, measurements, brand=None):
        """
        Backward-compatible method.

        If brand is supplied, return that brand's recommendation.
        Otherwise return the best brand and all relevant brands.
        """

        if brand:
            return self.recommend_brand(
                measurements,
                brand
            )

        all_brands = self.recommend_all_brands(
            measurements
        )

        if not all_brands:
            raise RuntimeError(
                "No brand recommendations available."
            )

        best = all_brands[0]

        return {
            **best,
            "brand_recommendations": all_brands,
        }
