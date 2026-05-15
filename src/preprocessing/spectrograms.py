"""
Spectrogram Representation Module
===================================

Converts raw 1-D PPG signal windows into 2-D spectrogram images using
Short-Time Fourier Transform (STFT). This vision-based representation
enables the CNN layers to exploit time-frequency patterns — heart rate
manifests as a clear spectral peak whose position and intensity are
easier for convolutional filters to learn than raw amplitude alone.

The spectrograms are generated on-the-fly during data loading (no disk
overhead) and can optionally be saved as .png for visualisation or
transfer-learning with pretrained image models.

Usage:
    from src.preprocessing.spectrograms import PPGSpectrogramDataset

    dataset = PPGSpectrogramDataset(data_dir="data/experiment_dataset_16s",
                                     subject="S1")
    spectrogram, label = dataset[0]
    # spectrogram shape: (1, n_freq_bins, n_time_frames)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


def ppg_to_spectrogram(
    ppg_window: np.ndarray,
    fs: int = 100,
    nperseg: int = 128,
    noverlap: int = 96,
    channel: int = 0,
) -> np.ndarray:
    """
    Convert a single PPG window to a log-magnitude spectrogram.

    Args:
        ppg_window: Array of shape (1600, 4) — a 16-second, 4-channel window.
        fs:         Sampling frequency in Hz.
        nperseg:    STFT segment length (higher = better frequency resolution).
        noverlap:   Overlap between consecutive segments.
        channel:    Which channel to transform (0 = filtered PPG).

    Returns:
        Log-magnitude spectrogram as a 2-D numpy array of shape
        (n_freq_bins, n_time_frames).
    """
    from scipy.signal import spectrogram as scipy_spectrogram

    signal = ppg_window[:, channel]
    freqs, times, Sxx = scipy_spectrogram(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap, mode="magnitude"
    )

    # Log scale (add epsilon to avoid log(0))
    log_spectrogram = np.log1p(Sxx)

    # Normalise to [0, 1]
    smin, smax = log_spectrogram.min(), log_spectrogram.max()
    if smax - smin > 1e-8:
        log_spectrogram = (log_spectrogram - smin) / (smax - smin)

    return log_spectrogram


def save_spectrogram_image(
    spectrogram: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 4),
):
    """Save a spectrogram as a .png image for visualisation."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


class PPGSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that serves PPG windows as spectrogram images.

    Each item returns:
        spectrogram: Tensor of shape (1, H, W) — single-channel image
        label:       Scalar heart rate in BPM
    """

    def __init__(
        self,
        data_dir: str,
        subject: str,
        fs: int = 100,
        nperseg: int = 128,
        noverlap: int = 96,
        channel: int = 0,
    ):
        data_path = Path(data_dir) / subject / "data.npy"
        label_path = Path(data_dir) / subject / "labels.npy"
        self.data = np.load(data_path)      # (N, 1600, 4)
        self.labels = np.load(label_path)   # (N,)
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.channel = channel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        spec = ppg_to_spectrogram(
            window, fs=self.fs, nperseg=self.nperseg,
            noverlap=self.noverlap, channel=self.channel,
        )
        # Add channel dimension: (H, W) → (1, H, W)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spec_tensor, label


if __name__ == "__main__":
    """Quick demo: generate and save a sample spectrogram."""
    import sys

    data_dir = Path("data/experiment_dataset_16s")
    if not data_dir.exists():
        print("Run preprocess_experiment.py first.")
        sys.exit(1)

    subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not subjects:
        print("No subject directories found.")
        sys.exit(1)

    subj = subjects[0]
    data = np.load(data_dir / subj / "data.npy")
    print(f"Subject {subj}: {data.shape[0]} windows")

    spec = ppg_to_spectrogram(data[0])
    print(f"Spectrogram shape: {spec.shape}")

    out_path = f"results/visualizations/spectrogram_{subj}_sample.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_spectrogram_image(spec, out_path, title=f"PPG Spectrogram — {subj}")
    print(f"Saved to {out_path}")
