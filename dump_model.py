import torch
import torch.nn as nn
import torch.nn.functional as F


class DumbSignalModel(nn.Module):
    def __init__(self, n_classes=6, n_channels=2, n_samples=128):
        """
        Simple model to test the training pipeline.

        Args:
            n_classes (int): number of output classes
            n_channels (int): number of input channels (I/Q)
            n_samples (int): length of the input signal
        """
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=n_channels, out_channels=4, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)  # reduces T -> 1

        self.fc = nn.Linear(8, n_classes)  # 8 features -> n_classes

    def forward(self, x):
        """
        x: [B, C, T]
        returns: [B, n_classes]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # [B, 8, 1]
        x = x.view(x.size(0), -1)  # [B, 8]
        x = self.fc(x)  # [B, n_classes]
        return x
