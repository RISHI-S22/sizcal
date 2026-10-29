from pathlib import Path
import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "deepfashion"
    / "images"
    / "img"
)

OUTPUT_DIR = PROJECT_ROOT / "models" / "pose_samples"

MAX_IMAGES = 1000
TOP_RESULTS = 20

VISIBILITY_THRESHOLD = 0.70


# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


REQUIRED_LANDMARKS = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
}


def calculate_score(landmarks):
    """
    Calculate average visibility of the landmarks required
    by the SIZCAL measurement pipeline.
    """

    scores = []

    for index in REQUIRED_LANDMARKS.values():
        scores.append(landmarks[index].visibility)

    return sum(scores) / len(scores)


def count_good_landmarks(landmarks):
    """
    Count required landmarks whose visibility exceeds
    the configured threshold.
    """

    count = 0

    for index in REQUIRED_LANDMARKS.values():
        if landmarks[index].visibility >= VISIBILITY_THRESHOLD:
            count += 1

    return count


def main():

    print("=" * 70)
    print("SIZCAL - FIND GOOD MEDIAPIPE POSE SAMPLES")
    print("=" * 70)

    print()
    print(f"Image root:")
    print(IMAGE_ROOT)

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"DeepFashion image directory not found:\n{IMAGE_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Collecting images...")

    image_paths = sorted(
        list(IMAGE_ROOT.rglob("*.jpg"))
        + list(IMAGE_ROOT.rglob("*.jpeg"))
        + list(IMAGE_ROOT.rglob("*.png"))
    )

    if not image_paths:
        raise RuntimeError("No images found.")

    image_paths = image_paths[:MAX_IMAGES]

    print(f"Images selected for scan: {len(image_paths)}")
    print()

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    results = []

    detected_count = 0

    try:

        for i, image_path in enumerate(image_paths, start=1):

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            result = pose.process(rgb)

            if result.pose_landmarks is None:
                continue

            detected_count += 1

            landmarks = result.pose_landmarks.landmark

            score = calculate_score(
                landmarks
            )

            good_landmarks = count_good_landmarks(
                landmarks
            )

            results.append(
                {
                    "score": score,
                    "good_landmarks": good_landmarks,
                    "path": image_path,
                }
            )

            if i % 50 == 0 or i == len(image_paths):

                print(
                    f"Processed: {i:5d}/{len(image_paths):5d} "
                    f"({i / len(image_paths) * 100:6.2f}%) | "
                    f"Poses: {detected_count:4d} | "
                    f"Candidates: {len(results):4d}"
                )

    finally:

        pose.close()

    print()
    print("=" * 70)
    print("SCAN COMPLETE")
    print("=" * 70)

    print()
    print(f"Images scanned:    {len(image_paths)}")
    print(f"Poses detected:    {detected_count}")
    print(f"Candidates found:  {len(results)}")

    if not results:

        print()
        print("No usable pose images were found.")
        print(
            "Increase MAX_IMAGES or lower "
            "VISIBILITY_THRESHOLD."
        )

        return

    results.sort(
        key=lambda item: (
            item["good_landmarks"],
            item["score"],
        ),
        reverse=True,
    )

    top_results = results[:TOP_RESULTS]

    print()
    print("=" * 70)
    print("BEST POSE CANDIDATES")
    print("=" * 70)

    for rank, item in enumerate(
        top_results,
        start=1,
    ):

        print()
        print(f"#{rank}")
        print(
            f"Visibility score : "
            f"{item['score']:.4f}"
        )
        print(
            f"Good landmarks   : "
            f"{item['good_landmarks']}/8"
        )
        print(
            f"Image            : "
            f"{item['path']}"
        )

    # Copy/save annotated versions of the top candidates.
    print()
    print("=" * 70)
    print("SAVING TOP CANDIDATES")
    print("=" * 70)

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:

        for rank, item in enumerate(
            top_results,
            start=1,
        ):

            image_path = item["path"]

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            result = pose.process(rgb)

            if result.pose_landmarks is None:
                continue

            annotated = image.copy()

            mp_pose.PoseLandmark

            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            output_path = (
                OUTPUT_DIR
                / f"pose_candidate_{rank:02d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                annotated,
            )

            print(output_path)

    finally:

        pose.close()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "Open the files in:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()