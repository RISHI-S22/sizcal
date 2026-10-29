from collections import Counter
from pathlib import Path

from backend.cnn.deepfashion_dataset import DeepFashionDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "deepfashion"
    / "images"
)

ANNOTATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "deepfashion"
    / "annotations"
    / "Category and Attribute Prediction Benchmark"
)


EXPECTED_TOTAL = 289222

EXPECTED_SPLITS = {
    "train": 209222,
    "val": 40000,
    "test": 40000,
}


def main():

    print("=" * 70)
    print("SIZCAL - DeepFashion Dataset Verification")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nImage root:")
    print(IMAGE_ROOT)

    print("\nAnnotation root:")
    print(ANNOTATION_ROOT)

    # ---------------------------------------------------------
    # Directory checks
    # ---------------------------------------------------------

    print("\nChecking directories...")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{IMAGE_ROOT}"
        )

    if not ANNOTATION_ROOT.exists():
        raise FileNotFoundError(
            f"Annotation directory does not exist:\n"
            f"{ANNOTATION_ROOT}"
        )

    print("Directories: OK")

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    print("\nLoading DeepFashion splits...")

    datasets = {}

    for split in ["train", "val", "test"]:

        print(f"\nLoading {split}...")

        dataset = DeepFashionDataset(
            root_dir=str(IMAGE_ROOT),
            annotation_dir=str(ANNOTATION_ROOT),
            split=split,
        )

        datasets[split] = dataset

        print(
            f"{split}: "
            f"{len(dataset):,} samples"
        )

        expected = EXPECTED_SPLITS[split]

        if len(dataset) != expected:

            raise RuntimeError(
                f"{split} split mismatch: "
                f"expected {expected:,}, "
                f"found {len(dataset):,}"
            )

    # ---------------------------------------------------------
    # Total image count
    # ---------------------------------------------------------

    total = sum(
        len(dataset)
        for dataset in datasets.values()
    )

    print("\nChecking total image count...")

    print(
        f"Expected total: {EXPECTED_TOTAL:,}"
    )

    print(
        f"Actual total:   {total:,}"
    )

    if total != EXPECTED_TOTAL:
        raise RuntimeError(
            "Total DeepFashion image count does not match "
            "289,222."
        )

    print("Total image count: OK")

    # ---------------------------------------------------------
    # Category information
    # ---------------------------------------------------------

    print("\nAnalyzing category IDs...")

    train_dataset = datasets["train"]

    category_mapping = (
        train_dataset.get_category_mapping()
    )

    reverse_mapping = (
        train_dataset.get_reverse_category_mapping()
    )

    category_ids = sorted(
        category_mapping.keys()
    )

    print(
        f"\nObserved category IDs: "
        f"{category_ids}"
    )

    print(
        f"\nObserved category count: "
        f"{len(category_ids)}"
    )

    print(
        "\nPyTorch class mapping:"
    )

    for pytorch_index in sorted(reverse_mapping):

        original_id = reverse_mapping[
            pytorch_index
        ]

        print(
            f"  PyTorch class {pytorch_index:02d}"
            f" -> DeepFashion category "
            f"{original_id}"
        )

    # ---------------------------------------------------------
    # Check labels
    # ---------------------------------------------------------

    print("\nChecking PyTorch labels...")

    all_labels = []

    for split, dataset in datasets.items():

        labels = []

        for i in range(len(dataset)):

            label = dataset.get_label(i)

            labels.append(label)

        all_labels.extend(labels)

        unique_labels = sorted(
            set(labels)
        )

        print(
            f"{split}: "
            f"minimum={min(labels)}, "
            f"maximum={max(labels)}, "
            f"classes={len(unique_labels)}"
        )

    unique_labels = sorted(
        set(all_labels)
    )

    expected_label_range = list(
        range(len(category_ids))
    )

    if unique_labels != expected_label_range:

        raise RuntimeError(
            "PyTorch labels are not contiguous.\n"
            f"Expected: {expected_label_range}\n"
            f"Found:    {unique_labels}"
        )

    print(
        f"PyTorch labels: "
        f"0-{len(category_ids) - 1}"
    )

    print(
        f"Number of actual classes: "
        f"{len(category_ids)}"
    )

    print("Label mapping: OK")

    # ---------------------------------------------------------
    # Category distribution
    # ---------------------------------------------------------

    print("\nChecking category distribution...")

    category_counts = Counter()

    for dataset in datasets.values():

        for i in range(len(dataset)):

            original_category = (
                dataset.get_original_category_id(i)
            )

            category_counts[
                original_category
            ] += 1

    print(
        "\nCategory image counts:"
    )

    for category_id in sorted(category_counts):

        count = category_counts[
            category_id
        ]

        pytorch_index = category_mapping[
            category_id
        ]

        print(
            f"  DeepFashion {category_id:02d}"
            f" -> PyTorch {pytorch_index:02d}"
            f" -> {count:,} images"
        )

    # ---------------------------------------------------------
    # Check image files
    # ---------------------------------------------------------

    print("\nChecking image files...")

    missing = []

    checked = 0

    for split, dataset in datasets.items():

        print(
            f"\nChecking {split} "
            f"({len(dataset):,} images)..."
        )

        for i in range(len(dataset)):

            image_path = (
                dataset.get_image_path(i)
            )

            if not image_path.exists():

                missing.append(
                    str(image_path)
                )

                if len(missing) >= 20:
                    break

            checked += 1

            if checked % 25000 == 0:

                percentage = (
                    checked / total
                ) * 100

                print(
                    f"  Progress: "
                    f"{percentage:.1f}% "
                    f"({checked:,}/{total:,})"
                )

        if missing:
            break

    print(
        f"\nFiles checked: "
        f"{checked:,}"
    )

    print(
        f"Missing images: "
        f"{len(missing)}"
    )

    if missing:

        print("\nFirst missing files:")

        for path in missing[:20]:
            print(path)

        raise RuntimeError(
            "Dataset verification failed because "
            "images are missing."
        )

    print("All referenced images exist.")

    # ---------------------------------------------------------
    # Test image loading
    # ---------------------------------------------------------

    print("\nTesting actual image loading...")

    for split, dataset in datasets.items():

        image, label = dataset[0]

        print(
            f"{split}: "
            f"image shape={tuple(image.shape)}, "
            f"label={label}"
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEEPFASHION VERIFICATION COMPLETE")
    print("=" * 70)

    print(
        f"Train images : "
        f"{len(datasets['train']):,}"
    )

    print(
        f"Val images   : "
        f"{len(datasets['val']):,}"
    )

    print(
        f"Test images  : "
        f"{len(datasets['test']):,}"
    )

    print(
        f"Total        : "
        f"{total:,}"
    )

    print(
        f"Actual classes: "
        f"{len(category_ids)}"
    )

    print(
        f"Missing      : "
        f"{len(missing)}"
    )

    print(
        "\nOfficial DeepFashion benchmark: "
        "50 categories"
    )

    print(
        "Actual category IDs present in the "
        "downloaded image annotation:"
    )

    print(category_ids)

    print(
        "\nThe PyTorch dataset uses a contiguous "
        "class mapping for the categories actually "
        "present in the annotation."
    )

    print(
        "\nDataset verification PASSED."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()