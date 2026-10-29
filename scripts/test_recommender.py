from backend.recommender import (
    SIZCALRecommender,
)


def main():

    print("=" * 70)
    print("SIZCAL - RECOMMENDER INTEGRATION TEST")
    print("=" * 70)

    recommender = (
        SIZCALRecommender()
    )

    print()
    print("Available brands:")

    for brand in recommender.get_brands():
        print(f"  - {brand}")

    print()
    print("Available sizes:")

    print(
        "  "
        + ", ".join(
            recommender.get_sizes()
        )
    )

    # -----------------------------------------------------
    # Test measurement vector
    # -----------------------------------------------------

    measurements = {
        "shoulder_width_cm": 44.0,
        "chest_circumference_cm": 100.0,
        "waist_circumference_cm": 84.0,
        "hip_circumference_cm": 100.0,
        "inseam_cm": 80.0,
    }

    print()
    print("TEST MEASUREMENTS")
    print("-" * 70)

    for name, value in measurements.items():

        print(
            f"{name:30s}: "
            f"{value:.2f} cm"
        )

    # -----------------------------------------------------
    # Test without brand
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TEST 1 - MODEL RECOMMENDATION")
    print("=" * 70)

    result = recommender.recommend(
        measurements
    )

    print()
    print(
        f"Recommended size: "
        f"{result['recommended_size']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence_percent']:.2f}%"
    )

    print()
    print("Ranked recommendations:")

    for item in result[
        "probabilities"
    ]:

        print(
            f"  {item['size']:3s} "
            f"{item['confidence_percent']:6.2f}%"
        )

    # -----------------------------------------------------
    # Test with brand
    # -----------------------------------------------------

    brand = recommender.get_brands()[0]

    print()
    print("=" * 70)
    print("TEST 2 - BRAND-AWARE RECOMMENDATION")
    print("=" * 70)

    print()
    print(f"Selected brand: {brand}")

    result = recommender.recommend(
        measurements,
        brand=brand,
    )

    print()
    print(
        f"Recommended size: "
        f"{result['recommended_size']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence_percent']:.2f}%"
    )

    print()
    print("Brand size chart:")

    for size, values in result[
        "brand_size_chart"
    ].items():

        print(
            f"  {size}: {values}"
        )

    print()
    print("=" * 70)
    print("RECOMMENDER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()