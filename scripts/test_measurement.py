from pathlib import Path

import cv2

from backend.measurement import (
    SIZCALMeasurementEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "deepfashion"
    / "images"
)


def find_test_image():

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ]

    for extension in extensions:

        files = list(
            IMAGE_ROOT.rglob(extension)
        )

        if files:
            return files[0]

    raise FileNotFoundError(
        "No image found in DeepFashion."
    )


def main():

    print()
    print("=" * 70)
    print("SIZCAL STAGE 2 - MEDIAPIPE MEASUREMENT TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = find_test_image()

    print()
    print(
        f"Test image:\n{image_path}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            "Unable to load test image."
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    # --------------------------------------------------------
    # Create measurement engine
    # --------------------------------------------------------

    engine = (
        SIZCALMeasurementEngine()
    )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is only a software smoke test.
        #
        # We use a temporary reference calibration so we can
        # verify the complete pipeline.
        #
        # The real application will obtain the reference
        # length from the user's calibration object.
        # ----------------------------------------------------

        reference_pixel_length = 200.0

        reference_length_cm = 10.0

        print()
        print(
            "Reference calibration:"
        )

        print(
            f"  Pixel length: "
            f"{reference_pixel_length}"
        )

        print(
            f"  Physical length: "
            f"{reference_length_cm} cm"
        )

        # ----------------------------------------------------
        # Run measurement
        # ----------------------------------------------------

        result = engine.measure(
            image=image,
            reference_pixel_length=(
                reference_pixel_length
            ),
            reference_length_cm=(
                reference_length_cm
            ),
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            "MEDIAPIPE RESULT"
        )

        print(
            "=" * 70
        )

        print()
        print(
            f"Pose detected: "
            f"{result['success']}"
        )

        print(
            f"Landmarks visible: "
            f"{result['landmarks_visible']}"
        )

        print(
            f"Scale: "
            f"{result['scale_cm_per_pixel']:.6f} "
            f"cm/pixel"
        )

        print()
        print(
            "MEASUREMENTS"
        )

        print(
            "-" * 70
        )

        for name, value in (
            result["measurements_cm"]
            .items()
        ):

            if value is None:

                print(
                    f"{name:35s}: N/A"
                )

            else:

                print(
                    f"{name:35s}: "
                    f"{value:.2f} cm"
                )

        print()
        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        pose_result = (
            engine.detect_pose(
                image
            )
        )

        annotated = (
            engine.draw_landmarks(
                image,
                pose_result,
            )
        )

        output_path = (
            PROJECT_ROOT
            / "models"
            / "measurement_test.jpg"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        print()
        print(
            "Annotated image saved:"
        )

        print(
            output_path
        )

    finally:

        engine.close()


if __name__ == "__main__":
    main()