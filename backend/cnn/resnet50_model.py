"""
SIZCAL ResNet-50 CNN

Deep-learning component of the SIZCAL pipeline.

Purpose:
- Load a pretrained ResNet-50 CNN using PyTorch.
- Use transfer learning for fashion-image representation.
- Provide image feature extraction.
- Provide a trainable classification head for SIZCAL.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class SIZCALResNet50(nn.Module):
    """
    ResNet-50 based CNN for SIZCAL.

    The pretrained ResNet-50 backbone is used to learn
    visual features from clothing/fashion images.

    The final fully connected layer is replaced with a
    SIZCAL-specific classification layer.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        # ResNet-50 produces a 2048-dimensional feature vector.
        feature_dim = self.backbone.fc.in_features

        # Remove the original ImageNet classification layer.
        self.backbone.fc = nn.Identity()

        # Optionally freeze the pretrained CNN.
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # SIZCAL classification head.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, num_classes),
        )

        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dimensional visual features.
        """
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns class logits.
        """
        features = self.extract_features(images)
        return self.classifier(features)


def get_image_transform():
    """
    Image preprocessing used by the ResNet-50 CNN.
    """

    weights = models.ResNet50_Weights.DEFAULT

    return weights.transforms()


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image and convert it to RGB.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return Image.open(image_path).convert("RGB")


def image_to_tensor(image_path: str | Path) -> torch.Tensor:
    """
    Load and preprocess an image for ResNet-50.

    Returns:
        Tensor with shape:
        [1, 3, H, W]
    """

    image = load_image(image_path)
    transform = get_image_transform()

    tensor = transform(image)

    return tensor.unsqueeze(0)


def extract_image_features(
    model: SIZCALResNet50,
    image_path: str | Path,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Extract a 2048-dimensional feature vector from an image.
    """

    model = model.to(device)
    model.eval()

    image_tensor = image_to_tensor(image_path).to(device)

    with torch.no_grad():
        features = model.extract_features(image_tensor)

    return features.cpu()


def save_model(
    model: SIZCALResNet50,
    output_path: str | Path,
):
    """
    Save SIZCAL ResNet-50 weights.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": model.feature_dim,
            "num_classes": model.num_classes,
        },
        output_path,
    )


def load_model(
    model_path: str | Path,
    num_classes: int = 10,
    device: str = "cpu",
) -> SIZCALResNet50:
    """
    Load a previously saved SIZCAL ResNet-50 model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = SIZCALResNet50(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model