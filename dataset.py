import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import math

class SignalsDataset(Dataset):
    def __init__(self, path_to_data, transform=None, magnitude_only=True, window_size=256):
        """
        Dataset PyTorch for signalsfrom a HDF5 file.
        we keep labels in int8, snr in int16.
        """
        super().__init__()
        self.path = path_to_data
        self.transform = transform
        self.magnitude_only = magnitude_only
        self.nfft = window_size

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
        signal = random_rotate_iq(signal) # Data augmentation: random IQ rotation
        if self.transform == "stft":
            spec = torch.stft(
                signal,
                n_fft=self.nfft,
                hop_length=self.nfft//2,
                win_length=self.nfft,
                window=torch.hann_window(self.nfft, device=signal.device),
                return_complex=True,
            )  # (C, F, T)
            if self.magnitude_only:
                signal = torch.abs(spec)  # (C, F, T)
            else:
                spec = torch.view_as_real(spec)  # (C, F, T, 2)
                spec = spec.permute(0, 2, 1, 3).contiguous()
                signal = spec.view(spec.size(0), spec.size(1), -1)
        label = torch.tensor(label, dtype=torch.int8)  # keep it as int8
        snr = torch.tensor(snr, dtype=torch.float32)  # float32 for model input

        return signal, label, snr


def rotate_iq(signal, angle_rad):
    """
    Rotate IQ signal by a given angle in radians.
    signal: tensor of shape [2, L] (I, Q)
    angle_rad: rotation angle in radians
    """
    I = signal[0]
    Q = signal[1]
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    I_rot = I * cos_theta - Q * sin_theta
    Q_rot = I * sin_theta + Q * cos_theta
    rotated = torch.stack([I_rot, Q_rot], dim=0)
    return rotated

def random_rotate_iq(signal):
    angle = 2 * math.pi * torch.rand(1).item()  # random angle 0 -> 2π
    return rotate_iq(signal, angle)