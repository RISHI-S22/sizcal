from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.cnn.deepfashion_dataset import DeepFashionDataset
from backend.cnn.resnet50_model import SIZCALResNet50


# ============================================================
# PROJECT PATHS
# ============================================================

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50_deepfashion_best.pth"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "models"
    / "features"
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 4

IMAGE_SIZE = 224

SPLITS = [
    "train",
    "val",
    "test",
]


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_trained_model(
    device,
    num_classes,
):
    """
    Load the trained SIZCAL ResNet-50 model.

    The training checkpoint currently contains keys such as:

        conv1.*
        bn1.*
        layer1.*
        ...
        fc.*

    The current SIZCALResNet50 wrapper expects:

        backbone.conv1.*
        backbone.bn1.*
        backbone.layer1.*
        ...
        classifier.*

    Therefore the checkpoint keys are translated before loading.
    """

    print()
    print("=" * 70)
    print("LOADING TRAINED RESNET-50")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nTrained model not found:\n{MODEL_PATH}\n\n"
            "Make sure the ResNet-50 training step has completed."
        )

    print(f"Model checkpoint:")
    print(MODEL_PATH)

    print()
    print("Creating SIZCALResNet50 model...")

    model = SIZCALResNet50(
        num_classes=num_classes
    )

    print("Model created.")

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # Determine whether the checkpoint is:
    #
    # 1. Direct state_dict
    # 2. A training checkpoint containing model_state_dict
    # 3. A training checkpoint containing state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    print(
        f"Checkpoint parameters: "
        f"{len(state_dict):,}"
    )

    # --------------------------------------------------------
    # Remove DataParallel prefix if present.
    #
    # Example:
    #
    # module.conv1.weight
    #
    # becomes:
    #
    # conv1.weight
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Translate training checkpoint keys to the current
    # SIZCALResNet50 wrapper.
    # --------------------------------------------------------

    translated_state_dict = {}

    for key, value in cleaned_state_dict.items():

        # Training classifier:
        #
        # fc.0.weight
        # fc.0.bias
        # fc.3.weight
        # fc.3.bias
        #
        # Current classifier:
        #
        # classifier.0.weight
        # classifier.0.bias
        # classifier.3.weight
        # classifier.3.bias

        if key.startswith("fc."):
            new_key = (
                "classifier."
                + key[3:]
            )

        # All torchvision ResNet backbone parameters
        # are placed under the current wrapper's backbone.
        else:
            new_key = (
                "backbone."
                + key
            )

        translated_state_dict[
            new_key
        ] = value

    print()
    print("Translating checkpoint parameter names...")

    # --------------------------------------------------------
    # Strict loading is intentional.
    #
    # We do NOT use strict=False because we want to make sure
    # the complete trained model has been loaded.
    # --------------------------------------------------------

    incompatible = model.load_state_dict(
        translated_state_dict,
        strict=True,
    )

    print()
    print("Checkpoint weights loaded successfully.")

    if incompatible.missing_keys:
        raise RuntimeError(
            "Unexpected missing model keys:\n"
            + "\n".join(
                incompatible.missing_keys
            )
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n"
            + "\n".join(
                incompatible.unexpected_keys
            )
        )

    model = model.to(device)

    model.eval()

    print(
        f"Model moved to: {device}"
    )

    return model


# ============================================================
# DATA LOADER
# ============================================================

def create_dataset(
    split,
):
    print()
    print(
        f"Loading DeepFashion {split} dataset..."
    )

    dataset = DeepFashionDataset(
        root_dir=str(IMAGE_ROOT),
        annotation_dir=str(ANNOTATION_ROOT),
        split=split,
    )

    print(
        f"{split} samples: "
        f"{len(dataset):,}"
    )

    print(
        f"Number of classes: "
        f"{dataset.num_classes}"
    )

    return dataset


def create_dataloader(
    dataset,
    batch_size,
    workers,
    device,
):
    pin_memory = (
        device.type == "cuda"
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }

    # persistent_workers only works when
    # num_workers > 0.
    if workers > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True

    loader = DataLoader(
        **loader_kwargs
    )

    return loader


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_features(
    model,
    dataset,
    loader,
    device,
    split,
):
    print()
    print("=" * 70)
    print(
        f"EXTRACTING {split.upper()} FEATURES"
    )
    print("=" * 70)

    total_images = len(dataset)

    print(
        f"Total images: "
        f"{total_images:,}"
    )

    print(
        f"Expected feature dimension: "
        f"2048"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_features = []
    all_labels = []
    all_category_ids = []
    all_image_names = []

    processed = 0

    start_time = time.time()

    # --------------------------------------------------------
    # Extraction loop
    # --------------------------------------------------------

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        images, labels = batch

        if device.type == "cuda":

            images = images.to(
                device,
                non_blocking=True,
            )

        else:

            images = images.to(
                device
            )

        # ----------------------------------------------------
        # Extract 2048-dimensional ResNet features.
        # ----------------------------------------------------

        features = model.extract_features(
            images
        )

        features = (
            features
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        labels_np = (
            labels
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        # ----------------------------------------------------
        # Store batch
        # ----------------------------------------------------

        all_features.append(
            features
        )

        all_labels.append(
            labels_np
        )

        # ----------------------------------------------------
        # Get original DeepFashion metadata.
        # ----------------------------------------------------

        batch_start = processed
        batch_end = (
            processed
            + len(labels_np)
        )

        for index in range(
            batch_start,
            batch_end,
        ):

            all_category_ids.append(
                dataset.get_original_category_id(
                    index
                )
            )

            all_image_names.append(
                dataset.get_image_name(
                    index
                )
            )

        processed = batch_end

        # ----------------------------------------------------
        # Progress information
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        percent = (
            processed
            / total_images
            * 100.0
        )

        images_per_second = (
            processed / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = (
            total_images
            - processed
        )

        eta_seconds = (
            remaining
            / images_per_second
            if images_per_second > 0
            else 0
        )

        eta_minutes = (
            eta_seconds / 60.0
        )

        if device.type == "cuda":

            memory_used = (
                torch.cuda.memory_allocated(
                    device
                )
                / (1024 ** 3)
            )

            memory_reserved = (
                torch.cuda.memory_reserved(
                    device
                )
                / (1024 ** 3)
            )

            gpu_text = (
                f" | VRAM "
                f"{memory_used:.2f}/"
                f"{memory_reserved:.2f} GB"
            )

        else:

            gpu_text = ""

        print(
            f"\r"
            f"{percent:6.2f}% | "
            f"{processed:,}/"
            f"{total_images:,} images | "
            f"{images_per_second:7.1f} img/s | "
            f"ETA {eta_minutes:6.1f} min"
            f"{gpu_text}",
            end="",
            flush=True,
        )

    print()

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    features_array = np.concatenate(
        all_features,
        axis=0,
    )

    labels_array = np.concatenate(
        all_labels,
        axis=0,
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        features_array.shape[0]
        != total_images
    ):
        raise RuntimeError(
            "Feature count does not match "
            "dataset size."
        )

    if (
        features_array.shape[1]
        != 2048
    ):
        raise RuntimeError(
            "Unexpected feature dimension: "
            f"{features_array.shape}"
        )

    if len(all_category_ids) != total_images:
        raise RuntimeError(
            "Category metadata count does not "
            "match dataset size."
        )

    if len(all_image_names) != total_images:
        raise RuntimeError(
            "Image metadata count does not "
            "match dataset size."
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Feature extraction complete."
    )

    print(
        f"Images processed: "
        f"{total_images:,}"
    )

    print(
        f"Feature shape: "
        f"{features_array.shape}"
    )

    print(
        f"Elapsed time: "
        f"{elapsed / 60:.2f} minutes"
    )

    return (
        features_array,
        labels_array,
        all_category_ids,
        all_image_names,
    )


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    split,
    features,
    labels,
    category_ids,
    image_names,
    dataset,
):
    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        FEATURE_ROOT
        / f"deepfashion_{split}_features.npz"
    )

    metadata_path = (
        FEATURE_ROOT
        / f"deepfashion_{split}_metadata.json"
    )

    # --------------------------------------------------------
    # Save numerical features
    # --------------------------------------------------------

    print()
    print(
        f"Saving features:"
    )

    print(
        feature_path
    )

    np.savez_compressed(
        feature_path,
        features=features,
        labels=labels,
        category_ids=np.asarray(
            category_ids,
            dtype=np.int64,
        ),
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "split": split,
        "num_samples": int(
            len(features)
        ),
        "feature_dimension": int(
            features.shape[1]
        ),
        "num_classes": int(
            dataset.num_classes
        ),
        "category_mapping": {
            str(k): int(v)
            for k, v
            in dataset.get_category_mapping().items()
        },
        "reverse_category_mapping": {
            str(k): int(v)
            for k, v
            in dataset.get_reverse_category_mapping().items()
        },
        "image_names": image_names,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Saved metadata:"
    )

    print(
        metadata_path
    )

    # --------------------------------------------------------
    # Report file sizes
    # --------------------------------------------------------

    feature_size_mb = (
        feature_path.stat().st_size
        / (1024 ** 2)
    )

    metadata_size_mb = (
        metadata_path.stat().st_size
        / (1024 ** 2)
    )

    print()
    print(
        f"Feature file size: "
        f"{feature_size_mb:.2f} MB"
    )

    print(
        f"Metadata file size: "
        f"{metadata_size_mb:.2f} MB"
    )


# ============================================================
# EXTRACT ONE SPLIT
# ============================================================

def process_split(
    split,
    model,
    device,
    batch_size,
    workers,
):
    dataset = create_dataset(
        split
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )

    (
        features,
        labels,
        category_ids,
        image_names,
    ) = extract_features(
        model=model,
        dataset=dataset,
        loader=loader,
        device=device,
        split=split,
    )

    save_features(
        split=split,
        features=features,
        labels=labels,
        category_ids=category_ids,
        image_names=image_names,
        dataset=dataset,
    )

    return {
        "split": split,
        "samples": len(features),
        "feature_dimension": features.shape[1],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract 2048-dimensional ResNet-50 "
            "features from DeepFashion."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=[
            "train",
            "val",
            "test",
            "all",
        ],
        help=(
            "Dataset split to process."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Batch size for feature extraction."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of DataLoader workers."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SIZCAL DEEPFASHION FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print()
    print(
        f"Image root:\n"
        f"{IMAGE_ROOT}"
    )

    print()
    print(
        f"Annotation root:\n"
        f"{ANNOTATION_ROOT}"
    )

    print()
    print(
        f"Model checkpoint:\n"
        f"{MODEL_PATH}"
    )

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"DeepFashion image directory not found:\n"
            f"{IMAGE_ROOT}"
        )

    if not ANNOTATION_ROOT.exists():
        raise FileNotFoundError(
            f"DeepFashion annotation directory not found:\n"
            f"{ANNOTATION_ROOT}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained ResNet-50 checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        cuda_version = (
            torch.version.cuda
        )

        print(
            f"GPU: {gpu_name}"
        )

        print(
            f"CUDA runtime: "
            f"{cuda_version}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

        # Enable faster convolution algorithms
        # for fixed-size 224x224 images.
        torch.backends.cudnn.benchmark = True

    else:

        print(
            "WARNING: CUDA is not available."
        )

        print(
            "Feature extraction will run on CPU."
        )

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    print()
    print(
        f"Batch size: "
        f"{args.batch_size}"
    )

    print(
        f"DataLoader workers: "
        f"{args.workers}"
    )

    print(
        f"Feature dimension: "
        f"2048"
    )

    # --------------------------------------------------------
    # Find number of classes.
    #
    # We load the training dataset first because the trained
    # classifier contains 46 observed DeepFashion categories.
    # --------------------------------------------------------

    train_dataset = create_dataset(
        "train"
    )

    num_classes = (
        train_dataset.num_classes
    )

    print()
    print(
        f"Model classes: "
        f"{num_classes}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_trained_model(
        device=device,
        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # Determine requested splits
    # --------------------------------------------------------

    if args.split == "all":

        requested_splits = SPLITS

    else:

        requested_splits = [
            args.split
        ]

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    results = []

    total_start_time = time.time()

    for split in requested_splits:

        result = process_split(
            split=split,
            model=model,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )

        results.append(
            result
        )

    total_elapsed = (
        time.time()
        - total_start_time
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION FINISHED")
    print("=" * 70)

    print()

    for result in results:

        print(
            f"{result['split']:>5} : "
            f"{result['samples']:>10,} images | "
            f"{result['feature_dimension']} features"
        )

    print()
    print(
        f"Total time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print()
    print(
        "Feature files are located in:"
    )

    print(
        FEATURE_ROOT
    )

    print()
    print(
        "Expected files:"
    )

    for split in requested_splits:

        print(
            f"  deepfashion_{split}_features.npz"
        )

        print(
            f"  deepfashion_{split}_metadata.json"
        )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Use the extracted ResNet-50 features "
        "for the SIZCAL recommendation/XGBoost stage."
    )

    print()


if __name__ == "__main__":
    main()