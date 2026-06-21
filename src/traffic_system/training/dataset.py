"""
Dataset loader for the crop-classifier training scripts (helmet, seatbelt).

Expects a directory structure of:
    data_dir/
        compliant/       *.jpg, *.png  (helmet present / seatbelt present)
        non_compliant/   *.jpg, *.png  (helmet absent / seatbelt absent)

This same class is reused for both classifiers since they're the same kind
of binary image-classification problem — only the data_dir changes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from torch.utils.data import Dataset
from torchvision import transforms

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_LABEL_DIRS = {"compliant": 0, "non_compliant": 1}


class CropClassificationDataset(Dataset):
    def __init__(self, data_dir: str | Path, image_size: int, augment: bool = False):
        self._data_dir = Path(data_dir)
        self._samples: list[tuple[Path, int]] = []

        for label_name, label_id in _LABEL_DIRS.items():
            label_dir = self._data_dir / label_name
            if not label_dir.exists():
                raise FileNotFoundError(
                    f"Expected training data directory not found: {label_dir}. "
                    f"Training data must be organized as "
                    f"{self._data_dir}/compliant/ and {self._data_dir}/non_compliant/."
                )
            for path in label_dir.iterdir():
                if path.suffix.lower() in _VALID_EXTENSIONS:
                    self._samples.append((path, label_id))

        if len(self._samples) == 0:
            raise ValueError(f"No images found under {self._data_dir}")

        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            base_transforms += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(degrees=8),
            ]
        base_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self._transform = transforms.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise IOError(f"Failed to read image at {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image_rgb)
        return tensor, label

    @property
    def class_balance(self) -> dict[str, int]:
        counts = {"compliant": 0, "non_compliant": 0}
        for _, label in self._samples:
            name = "compliant" if label == 0 else "non_compliant"
            counts[name] += 1
        return counts
