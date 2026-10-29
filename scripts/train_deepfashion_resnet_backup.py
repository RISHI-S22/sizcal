from pathlib import Path
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from backend.cnn.deepfashion_dataset import DeepFashionDataset


# ============================================================
# PATHS
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

MODEL_ROOT = PROJECT_ROOT / "models"

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEFAULT TRAINING SETTINGS
# ============================================================

DEFAULT_EPOCHS = 10

# RTX 4060 8 GB
DEFAULT_BATCH_SIZE = 32

DEFAULT_LEARNING_RATE = 1e-4

DEFAULT_WEIGHT_DECAY = 1e-4

# Windows + RTX 4060
DEFAULT_NUM_WORKERS = 4

IMAGE_SIZE = 224

AMP_ENABLED = True


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print()
        print("=" * 70)
        print("CUDA GPU DETECTED")
        print("=" * 70)

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA runtime: {torch.version.cuda}"
        )

        print(
            f"GPU count: {torch.cuda.device_count()}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: {total_memory:.2f} GB"
        )

        print("=" * 70)

        return device

    print()
    print("=" * 70)
    print("CUDA GPU NOT AVAILABLE")
    print("Falling back to CPU.")
    print("=" * 70)

    return torch.device("cpu")


# ============================================================
# DATA TRANSFORMS
# ============================================================

def create_transforms():

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return (
        train_transform,
        validation_transform,
    )


# ============================================================
# DATASETS
# ============================================================

def create_datasets():

    print()
    print("=" * 70)
    print("LOADING DEEPFASHION DATASET")
    print("=" * 70)

    train_transform, validation_transform = (
        create_transforms()
    )

    train_dataset = DeepFashionDataset(
        root_dir=str(IMAGE_ROOT),
        annotation_dir=str(ANNOTATION_ROOT),
        split="train",
        transform=train_transform,
    )

    validation_dataset = DeepFashionDataset(
        root_dir=str(IMAGE_ROOT),
        annotation_dir=str(ANNOTATION_ROOT),
        split="val",
        transform=validation_transform,
    )

    print(
        f"Training images   : {len(train_dataset):,}"
    )

    print(
        f"Validation images : {len(validation_dataset):,}"
    )

    print(
        f"Number of classes : {train_dataset.num_classes}"
    )

    print("=" * 70)

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    validation_dataset,
    batch_size,
    num_workers,
    device,
):

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes):

    print()
    print("=" * 70)
    print("CREATING RESNET-50")
    print("=" * 70)

    weights = (
        models.ResNet50_Weights.DEFAULT
    )

    model = models.resnet50(
        weights=weights
    )

    original_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Linear(
            original_features,
            512,
        ),

        nn.ReLU(),

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            512,
            num_classes,
        ),
    )

    print(
        f"Original feature size : {original_features}"
    )

    print(
        f"Output classes        : {num_classes}"
    )

    print(
        "Pretrained weights    : ImageNet"
    )

    print("=" * 70)

    return model


# ============================================================
# GPU MEMORY
# ============================================================

def get_gpu_memory():

    if not torch.cuda.is_available():
        return None

    allocated = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )

    total = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    return (
        allocated,
        reserved,
        total,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    total_epochs,
    use_amp,
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    epoch_start = time.time()

    total_images = len(loader.dataset)

    processed_images = 0

    print()
    print(
        f"Epoch {epoch}/{total_epochs}"
    )

    print("-" * 70)

    for batch_index, (
        images,
        labels,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size_actual = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size_actual
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size_actual

        processed_images += (
            batch_size_actual
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        percent = (
            processed_images
            / total_images
            * 100
        )

        if processed_images > 0:

            images_per_second = (
                processed_images
                / elapsed
            )

            remaining_images = (
                total_images
                - processed_images
            )

            eta_seconds = (
                remaining_images
                / images_per_second
            )

        else:

            images_per_second = 0

            eta_seconds = 0

        avg_loss = (
            running_loss / total
        )

        accuracy = (
            correct
            / total
            * 100
        )

        eta_minutes = (
            eta_seconds / 60
        )

        gpu_text = ""

        gpu_memory = (
            get_gpu_memory()
        )

        if gpu_memory is not None:

            allocated, reserved, total_memory = (
                gpu_memory
            )

            gpu_text = (
                f" | VRAM: "
                f"{allocated:.2f}/"
                f"{total_memory:.2f} GB"
            )

        print(
            f"\r"
            f"Progress: {percent:6.2f}%"
            f" | Images: "
            f"{processed_images:,}/"
            f"{total_images:,}"
            f" | Loss: {avg_loss:.4f}"
            f" | Acc: {accuracy:6.2f}%"
            f" | Speed: "
            f"{images_per_second:6.1f} img/s"
            f" | ETA: "
            f"{eta_minutes:6.1f} min"
            f"{gpu_text}",
            end="",
            flush=True,
        )

    epoch_time = (
        time.time()
        - epoch_start
    )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct
        / total
        * 100
    )

    print()

    print(
        f"Epoch {epoch} complete"
    )

    print(
        f"Loss     : {epoch_loss:.4f}"
    )

    print(
        f"Accuracy : {epoch_accuracy:.2f}%"
    )

    print(
        f"Time     : "
        f"{epoch_time / 60:.2f} minutes"
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_time,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    epoch,
    total_epochs,
    use_amp,
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    start_time = time.time()

    total_images = len(loader.dataset)

    processed_images = 0

    print()
    print(
        f"Validation {epoch}/{total_epochs}"
    )

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        batch_size_actual = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size_actual
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size_actual

        processed_images += (
            batch_size_actual
        )

        percent = (
            processed_images
            / total_images
            * 100
        )

        print(
            f"\rValidation progress: "
            f"{percent:6.2f}%"
            f" | Images: "
            f"{processed_images:,}/"
            f"{total_images:,}",
            end="",
            flush=True,
        )

    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct
        / total
        * 100
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()

    print(
        f"Validation loss     : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation accuracy : "
        f"{validation_accuracy:.2f}%"
    )

    print(
        f"Validation time     : "
        f"{elapsed / 60:.2f} minutes"
    )

    return (
        validation_loss,
        validation_accuracy,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_accuracy,
    train_dataset,
    history,
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_accuracy":
            best_accuracy,

        "num_classes":
            train_dataset.num_classes,

        "category_to_index":
            train_dataset.get_category_mapping(),

        "index_to_category":
            train_dataset.get_reverse_category_mapping(),

        "history":
            history,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model(
    epochs,
    batch_size,
    learning_rate,
    num_workers,
):

    device = get_device()

    use_amp = (
        AMP_ENABLED
        and device.type == "cuda"
    )

    if device.type == "cuda":

        torch.backends.cudnn.benchmark = True

    print()
    print(
        f"Device: {device}"
    )

    print(
        f"AMP enabled: {use_amp}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"DataLoader workers: "
        f"{num_workers}"
    )

    (
        train_dataset,
        validation_dataset,
    ) = create_datasets()

    (
        train_loader,
        validation_loader,
    ) = create_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size,
        num_workers,
        device,
    )

    model = create_model(
        train_dataset.num_classes
    )

    model = model.to(device)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    history = []

    best_accuracy = 0.0

    total_start_time = time.time()

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        f"Epochs          : {epochs}"
    )

    print(
        f"Training images : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation      : "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Classes         : "
        f"{train_dataset.num_classes}"
    )

    print(
        f"Batch size      : "
        f"{batch_size}"
    )

    print(
        f"Learning rate   : "
        f"{learning_rate}"
    )

    print(
        f"AMP             : "
        f"{use_amp}"
    )

    print("=" * 70)

    for epoch in range(
        1,
        epochs + 1,
    ):

        (
            train_loss,
            train_accuracy,
            train_time,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            epochs,
            use_amp,
        )

        (
            validation_loss,
            validation_accuracy,
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
            epoch,
            epochs,
            use_amp,
        )

        scheduler.step(
            validation_accuracy
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_record = {

            "epoch": epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "validation_loss":
                validation_loss,

            "validation_accuracy":
                validation_accuracy,

            "learning_rate":
                current_lr,

            "train_time_seconds":
                train_time,
        }

        history.append(
            epoch_record
        )

        last_model_path = (
            MODEL_ROOT
            / "resnet50_deepfashion_last.pth"
        )

        save_checkpoint(
            last_model_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_accuracy,
            train_dataset,
            history,
        )

        if validation_accuracy > best_accuracy:

            best_accuracy = (
                validation_accuracy
            )

            best_model_path = (
                MODEL_ROOT
                / "resnet50_deepfashion_best.pth"
            )

            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_accuracy,
                train_dataset,
                history,
            )

            print()
            print(
                f"NEW BEST MODEL: "
                f"{best_accuracy:.2f}%"
            )

        history_path = (
            MODEL_ROOT
            / "resnet50_training_history.json"
        )

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch}/{epochs}"
        )

        print(
            f"Train accuracy : "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Val accuracy   : "
            f"{validation_accuracy:.2f}%"
        )

        print(
            f"Best accuracy  : "
            f"{best_accuracy:.2f}%"
        )

        print(
            f"Learning rate  : "
            f"{current_lr:.8f}"
        )

        print("=" * 70)

    total_time = (
        time.time()
        - total_start_time
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Total time: "
        f"{total_time / 3600:.2f} hours"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        MODEL_ROOT
        / "resnet50_deepfashion_best.pth"
    )

    print(
        MODEL_ROOT
        / "resnet50_deepfashion_last.pth"
    )

    print()
    print(
        "Training history:"
    )

    print(
        MODEL_ROOT
        / "resnet50_training_history.json"
    )

    print("=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train ResNet-50 on "
            "DeepFashion"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.workers,
    )