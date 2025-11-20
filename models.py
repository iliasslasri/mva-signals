import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTSignalModel(nn.Module):
    def __init__(self, n_classes=6, n_channels=2):
        super().__init__()
        assert n_channels == 2, "Model written for 2 channels"

        # groups=2 means: channel0 processed by weights0, channel1 by weights1
        self.conv1 = nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3, groups=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, groups=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2, groups=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, groups=2)

        # Adaptive pooling applied per channel
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Each channel outputs 16 → total 32 dims → FC
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: [B,2,H,W]
        x = F.relu(self.conv1(x))   # [B,32,H/2,W/2]   → 16 maps for ch0 + 16 maps for ch1
        x = F.relu(self.conv2(x))   # [B,64,...]       → still separated internally
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))   # [B,32,...] (16 features per channel)

        x = self.pool(x)            # [B,32,1,1]
        x = x.view(x.size(0), -1)   # [B,32] (16 per channel × 2)
        
        logits = self.fc(x)
        return logits   # DO NOT softmax here


class CNN_LSTM_SNR_Model(nn.Module):
    def __init__(self, n_classes=6, n_channels=2, hidden_size=64, snr_emb_dim=16, include_snr=True):
        super().__init__()
        self.include_snr = include_snr
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        if include_snr:
            self.snr_emb = nn.Linear(1, snr_emb_dim)  # embed scalar SNR
            self.fc1 = nn.Linear(hidden_size*2 + snr_emb_dim, 128)
        else:
            self.fc1 = nn.Linear(hidden_size*2, 128)

        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, snr):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [B, T, C] for LSTM
        x, _ = self.lstm(x)
        x = x.mean(dim=1)       # [B, hidden*2]
        if not self.include_snr:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        snr = snr.float()       # ensure float
        if len(snr.shape) == 1:
            snr = snr.unsqueeze(1)
        snr_feat = F.relu(self.snr_emb(snr))  # [B, snr_emb_dim]

        x = torch.cat([x, snr_feat], dim=1)  # [B, hidden*2 + snr_emb_dim]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

