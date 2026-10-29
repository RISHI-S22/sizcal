from pathlib import Path
import time

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_URL = "http://127.0.0.1:8000"

TEST_IMAGE = (
    PROJECT_ROOT
    / "models"
    / "pose_samples"
    / "pose_candidate_01.jpg"
)


def main():

    print("=" * 70)
    print("SIZCAL - FASTAPI END-TO-END TEST")
    print("=" * 70)

    print()
    print(f"Test image:\n{TEST_IMAGE}")

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(
            f"Test image not found:\n{TEST_IMAGE}\n\n"
            "Check the image path or use an existing DeepFashion image."
        )

    # -----------------------------------------------------
    # TEST 1: HEALTH
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TEST 1 - HEALTH")
    print("=" * 70)

    response = requests.get(
        f"{API_URL}/api/health",
        timeout=30,
    )

    print(f"HTTP status: {response.status_code}")
    print(response.json())

    if response.status_code != 200:
        raise RuntimeError("Health check failed.")

    # -----------------------------------------------------
    # TEST 2: BRANDS
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TEST 2 - BRANDS")
    print("=" * 70)

    response = requests.get(
        f"{API_URL}/api/brands",
        timeout=30,
    )

    print(f"HTTP status: {response.status_code}")
    print(response.json())

    if response.status_code != 200:
        raise RuntimeError("Brands endpoint failed.")

    # -----------------------------------------------------
    # TEST 3: END-TO-END ANALYSIS
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TEST 3 - END-TO-END ANALYSIS")
    print("=" * 70)

    # Software-test calibration only.
    # This is NOT real physical calibration.
    data = {
        "reference_pixel_length": "200",
        "reference_length_cm": "10",
        "brand": "brand_a",
    }

    start = time.time()

    with TEST_IMAGE.open("rb") as image_file:

        files = {
            "file": (
                TEST_IMAGE.name,
                image_file,
                "image/jpeg",
            )
        }

        response = requests.post(
            f"{API_URL}/api/analyze",
            data=data,
            files=files,
            timeout=300,
        )

    elapsed = time.time() - start

    print()
    print(f"HTTP status: {response.status_code}")
    print(f"Request time: {elapsed:.2f} seconds")

    result = response.json()

    print()
    print("Response:")
    print(result)

    if response.status_code != 200:
        raise RuntimeError(
            "End-to-end API request failed."
        )

    if not result.get("success", False):
        raise RuntimeError(
            "SIZCAL analysis returned success=False."
        )

    recommendation = result["recommendation"]

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print()
    print(
        "Recommended size:",
        recommendation["recommended_size"],
    )

    print(
        "Confidence:",
        recommendation["confidence_percent"],
        "%",
    )

    print()
    print("Ranked sizes:")

    for item in recommendation["probabilities"]:

        print(
            f"  {item['size']:3s} "
            f"{item['confidence_percent']:6.2f}%"
        )

    print()
    print("=" * 70)
    print("FASTAPI END-TO-END TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()