import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class STFT_CNN_LSTM_SNR_Model(nn.Module):
    def __init__(self, n_classes=6, n_channels=2, hidden_size=64, snr_emb_dim=16, include_snr=True):
        super().__init__()
        self.include_snr = include_snr
        assert n_channels == 2, "Designed for 2-channel STFT input"

        # CNN layers on STFT [B, 2, F, T]
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Pool only along frequency axis (preserve temporal info)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, T]

        # LSTM on temporal dimension
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)

        # Optional SNR embedding
        if include_snr:
            self.snr_emb = nn.Linear(1, snr_emb_dim)
            self.fc1 = nn.Linear(hidden_size*2 + snr_emb_dim, 128)
        else:
            self.fc1 = nn.Linear(hidden_size*2, 128)

        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, snr=None):
        # x: [B, 2, F, T] STFT input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Pool frequency axis only
        x = self.freq_pool(x)  # [B, C, 1, T]
        x = x.squeeze(2)       # [B, C, T]
        x = x.permute(0, 2, 1) # [B, T, C] for LSTM

        x, _ = self.lstm(x)
        x = x.mean(dim=1)      # [B, hidden*2]

        if self.include_snr:
            snr = snr.float().unsqueeze(1) if len(snr.shape) == 1 else snr
            snr_feat = F.relu(self.snr_emb(snr))
            x = torch.cat([x, snr_feat], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



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

