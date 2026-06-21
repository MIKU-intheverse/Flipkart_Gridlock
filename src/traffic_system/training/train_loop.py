"""
Shared training loop for the crop binary classifiers. Both
train_helmet_classifier.py and train_seatbelt_classifier.py call into this
with their own config section, rather than duplicating the training loop.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from traffic_system.utils.logging_utils import get_logger
from traffic_system.violations.crop_classifier_model import CropBinaryClassifier
from traffic_system.training.dataset import CropClassificationDataset

logger = get_logger(__name__)


def train_classifier(
    data_dir: str,
    output_path: str,
    image_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    device: torch.device,
) -> dict:
    """
    Trains a CropBinaryClassifier from scratch on the given data directory
    and saves the resulting state_dict to output_path. Returns a dict of
    final metrics so the calling script can log/report them.
    """
    full_dataset = CropClassificationDataset(data_dir, image_size=image_size, augment=True)
    logger.info("Loaded %d training images. Class balance: %s",
                len(full_dataset), full_dataset.class_balance)

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = CropBinaryClassifier(input_size=image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_set)
        val_accuracy, val_loss = _evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_accuracy=%.4f",
            epoch, epochs, train_loss, val_loss, val_accuracy,
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), output_path_obj)
            logger.info("Saved new best model (val_accuracy=%.4f) to %s", val_accuracy, output_path_obj)

    return {"best_val_accuracy": best_val_accuracy, "output_path": str(output_path_obj)}


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss
