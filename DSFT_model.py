import torch
import torch.nn as nn
import torch.nn.functional as F


class DSFTSignalModel(nn.Module):
    def __init__(self, n_classes=6, n_channels=2):
        """
        Progressive time-reducing 2D CNN.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)  # now T=1
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B,16,T/2]
        x = F.relu(self.conv2(x))  # [B,32,T/4]
        x = F.relu(self.conv3(x))  # [B,64,T/8]
        x = F.relu(self.conv4(x))  # [B,32,T/16]
        x = F.relu(self.conv5(x))  # [B,16,T/32]
        x = self.pool(x)  # [B,16,1]
        x = x.view(x.size(0), -1)  # [B,16]
        x = self.fc(x)  # [B,n_classes]
        x = F.softmax(x, dim=1)
        return x
