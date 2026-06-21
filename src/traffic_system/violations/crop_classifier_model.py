"""
Shared lightweight CNN binary classifier architecture, used by both the
helmet and seatbelt checks (and their training scripts). Kept as one small
model definition rather than two separate architectures, since both tasks
are the same shape of problem: classify a small cropped region as
compliant / non-compliant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CropBinaryClassifier(nn.Module):
    """
    A small CNN sized for fast inference on tiny crops (helmet/seatbelt
    regions are typically under 150px), not a heavyweight backbone like
    ResNet — matching the project plan's "lightweight classifier" scope.
    """

    def __init__(self, input_size: int = 96):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        reduced = input_size // 8  # after 3 pooling steps
        self.fc1 = nn.Linear(64 * reduced * reduced, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)   # 2 classes: compliant=0, non_compliant=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability of the non_compliant class (index 1)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]
