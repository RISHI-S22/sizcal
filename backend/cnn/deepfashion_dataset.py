from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepFashionDataset(Dataset):
    """
    PyTorch Dataset for the DeepFashion
    Category and Attribute Prediction Benchmark.

    Uses:
        - Anno_coarse/list_category_img.txt
        - Eval/list_eval_partition.txt

    Important:
        The official DeepFashion benchmark describes 50 clothing
        categories, but the actual category-image annotation file
        may contain only a subset of category IDs.

        Therefore, this class discovers the category IDs actually
        present in the annotation file and maps them to contiguous
        PyTorch labels:

            actual category ID -> 0..N-1

        This prevents gaps in CrossEntropyLoss labels.
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str,
        annotation_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.annotation_dir = Path(annotation_dir)
        self.split = split

        if split not in self.VALID_SPLITS:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        self.category_file = (
            self.annotation_dir
            / "Anno_coarse"
            / "list_category_img.txt"
        )

        self.partition_file = (
            self.annotation_dir
            / "Eval"
            / "list_eval_partition.txt"
        )

        if not self.category_file.exists():
            raise FileNotFoundError(
                f"Category annotation not found:\n"
                f"{self.category_file}"
            )

        if not self.partition_file.exists():
            raise FileNotFoundError(
                f"Partition annotation not found:\n"
                f"{self.partition_file}"
            )

        self.transform = (
            transform
            if transform is not None
            else self.default_transform()
        )

        self.category_labels = self._read_category_labels()
        self.partitions = self._read_partitions()

        self.category_ids = sorted(
            set(self.category_labels.values())
        )

        self.category_to_index = {
            category_id: index
            for index, category_id in enumerate(self.category_ids)
        }

        self.index_to_category = {
            index: category_id
            for category_id, index
            in self.category_to_index.items()
        }

        self.num_classes = len(self.category_ids)

        self.samples = self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}'. "
                "Check image and annotation paths."
            )

    @staticmethod
    def default_transform():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _read_category_labels(self):
        """
        Read:

            image_name category_id

        from list_category_img.txt.
        """

        labels = {}

        with self.category_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            first_line = f.readline().strip()

            if not first_line:
                raise RuntimeError(
                    "Category annotation file is empty."
                )

            expected_count = int(first_line)

            # Header
            f.readline()

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                image_name = parts[0]
                category_id = int(parts[1])

                labels[image_name] = category_id

        if len(labels) != expected_count:
            raise RuntimeError(
                "Category annotation count mismatch: "
                f"expected {expected_count}, "
                f"got {len(labels)}"
            )

        return labels

    def _read_partitions(self):
        """
        Read:

            image_name evaluation_status

        from list_eval_partition.txt.
        """

        partitions = {}

        with self.partition_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            first_line = f.readline().strip()

            if not first_line:
                raise RuntimeError(
                    "Partition annotation file is empty."
                )

            expected_count = int(first_line)

            # Header
            f.readline()

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                image_name = parts[0]
                evaluation_status = parts[1]

                partitions[image_name] = evaluation_status

        if len(partitions) != expected_count:
            raise RuntimeError(
                "Partition annotation count mismatch: "
                f"expected {expected_count}, "
                f"got {len(partitions)}"
            )

        return partitions

    def _load_samples(self):
        """
        Build samples for the requested split.

        Labels are converted from the original DeepFashion
        category IDs to contiguous 0..N-1 PyTorch class indices.
        """

        samples = []

        for image_name, category_id in self.category_labels.items():

            partition = self.partitions.get(image_name)

            if partition != self.split:
                continue

            image_path = self.root_dir / image_name

            class_index = self.category_to_index[
                category_id
            ]

            samples.append(
                (
                    image_path,
                    class_index,
                    image_name,
                    category_id,
                )
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path = self.samples[index][0]
        class_index = self.samples[index][1]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_index

    def get_image_path(self, index):
        return self.samples[index][0]

    def get_image_name(self, index):
        return self.samples[index][2]

    def get_label(self, index):
        return self.samples[index][1]

    def get_original_category_id(self, index):
        return self.samples[index][3]

    def get_category_mapping(self):
        """
        Returns:

            {
                original_deepfashion_category_id:
                    contiguous_pytorch_index
            }
        """

        return self.category_to_index

    def get_reverse_category_mapping(self):
        """
        Returns:

            {
                contiguous_pytorch_index:
                    original_deepfashion_category_id
            }
        """

        return self.index_to_category