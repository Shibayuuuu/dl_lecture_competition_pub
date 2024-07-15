import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
import scipy.signal
from sklearn.preprocessing import StandardScaler


def resample_data(X, new_rate):
    num_samples = X.shape[-1]
    resampled = scipy.signal.resample(X, new_rate, axis=-1)
    return torch.tensor(resampled, dtype=torch.float32)

def filter_data(X, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered = scipy.signal.lfilter(b, a, X, axis=-1)
    return torch.tensor(filtered, dtype=torch.float32)

def scale_data(X):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1]))
    return torch.tensor(scaled.reshape(X.shape), dtype=torch.float32)

def baseline_correction(X):
    baseline = X.mean(axis=-1, keepdims=True)
    corrected = X - baseline
    return corrected.clone().detach().float()


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

        # 前処理パラメータの初期化
        self.new_rate = 100
        self.lowcut = 0.1
        self.highcut = 30
        self.fs = 100

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path)).float()
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path)).long()
        
        # 前処理を適用
        X = resample_data(X, self.new_rate)
        X = filter_data(X, self.lowcut, self.highcut, self.fs)
        X = scale_data(X)
        X = baseline_correction(X)

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path)).long()
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]