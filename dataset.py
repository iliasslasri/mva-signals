import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SignalsDataset(Dataset):
    def __init__(self, path_to_data, transform=None):
        """
        Dataset PyTorch for signalsfrom a HDF5 file.
        we keep labels in int8, snr in int16.
        """
        super().__init__()
        self.path = path_to_data
        self.transform = transform

        with h5py.File(self.path, "r") as f:
            self.signals = np.array(f["signaux"], dtype=np.float32)  # (N, 2, L)
            self.labels = np.array(f["labels"], dtype=np.int8)  # (N,)
            self.snr = np.array(f["snr"], dtype=np.int16)  # (N,)

        print(
            f"Dataset loaded: {len(self.signals)} samples, each of shape {self.signals[0].shape}"
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]  # (2, L)
        label = self.labels[idx]  # int8
        snr = self.snr[idx]  # int16

        # Convertir en Tensor
        signal = torch.tensor(signal, dtype=torch.float32).transpose(
            -1, -2
        )  # (L, 2) -> (2, L)
        if self.transform == "stft":
            signal = torch.stft(
                signal,
                window=torch.hann_window(256),
                n_fft=256,
                hop_length=128,
                win_length=256,
                return_complex=True,
            )
            signal = torch.view_as_real(signal)
            signal = signal.permute(2, 0, 1, 3).contiguous()
            signal = signal.view(signal.size(0), signal.size(1), -1)
        label = torch.tensor(label, dtype=torch.int8)  # garde int8
        snr = torch.tensor(snr, dtype=torch.float32)

        return signal, label, snr
