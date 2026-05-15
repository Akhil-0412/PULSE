"""
DCGAN-based Synthetic PPG Data Generator
=========================================

Generates realistic synthetic PPG signal windows using a Deep Convolutional
Generative Adversarial Network (DCGAN). The generator learns the temporal
distribution of real PPG windows and produces new samples to augment the
training set — particularly useful for under-represented subjects or
motion-artifact-heavy segments.

Architecture follows Radford et al. (2016) with adaptations for 1-D
physiological signals:
  - Transposed 1-D convolutions in the Generator
  - Strided 1-D convolutions in the Discriminator
  - Spectral normalisation on the Discriminator for stable training

Usage:
    python -m src.augmentation.dcgan --data_dir data/experiment_dataset_16s \
                                      --epochs 200 --augment_ratio 0.15
"""

import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Generator: latent vector z → synthetic PPG window (4 channels × 1600)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Maps a latent vector to a 4-channel PPG window of length 1600.

    Architecture:
        z (128) → Linear → Reshape → ConvTranspose1d blocks → (4, 1600)
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 4):
        super().__init__()
        self.latent_dim = latent_dim

        # Project and reshape: z → (512, 25)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 512 * 25),
            nn.BatchNorm1d(512 * 25),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            # (512, 25) → (256, 50)
            nn.ConvTranspose1d(512, 256, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # (256, 50) → (128, 100)
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # (128, 100) → (64, 400)
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # (64, 400) → (4, 1600)
            nn.ConvTranspose1d(64, out_channels, kernel_size=8, stride=4, padding=2, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)          # (B, 512*25)
        x = x.view(-1, 512, 25)      # (B, 512, 25)
        x = self.conv_blocks(x)      # (B, 4, 1600)
        return x


# ---------------------------------------------------------------------------
# Discriminator: PPG window (4, 1600) → real / fake probability
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Binary classifier distinguishing real from synthetic PPG windows.

    Architecture:
        (4, 1600) → Strided Conv1d blocks → Linear → sigmoid
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            # (4, 1600) → (64, 400)
            nn.utils.spectral_norm(
                nn.Conv1d(in_channels, 64, kernel_size=8, stride=4, padding=2, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 400) → (128, 100)
            nn.utils.spectral_norm(
                nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 100) → (256, 50)
            nn.utils.spectral_norm(
                nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 50) → (512, 25)
            nn.utils.spectral_norm(
                nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=3, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 25, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_dcgan(
    data_dir: Path,
    epochs: int = 200,
    batch_size: int = 64,
    latent_dim: int = 128,
    lr: float = 2e-4,
    augment_ratio: float = 0.15,
    device: str = "auto",
):
    """
    Train DCGAN on real PPG windows, then generate synthetic samples.

    Args:
        data_dir:       Path to experiment_dataset_16s/
        epochs:         Number of GAN training epochs
        batch_size:     Mini-batch size
        latent_dim:     Dimensionality of the latent space
        lr:             Learning rate for both G and D
        augment_ratio:  Fraction of synthetic samples to generate relative to
                        the real dataset size (0.15 = 15% augmentation)
        device:         "cuda", "cpu", or "auto"
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[DCGAN] Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load all real training windows
    # ------------------------------------------------------------------
    all_data, all_labels = [], []
    subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    for subj in subjects:
        data_path = data_dir / subj / "data.npy"
        label_path = data_dir / subj / "labels.npy"
        if data_path.exists():
            d = np.load(data_path)          # (N, 1600, 4)
            l = np.load(label_path)
            all_data.append(d)
            all_labels.append(l)

    real_data = np.concatenate(all_data, axis=0)   # (Total, 1600, 4)
    real_labels = np.concatenate(all_labels, axis=0)
    n_real = len(real_data)
    print(f"[DCGAN] Loaded {n_real} real windows from {len(subjects)} subjects")

    # Transpose to (N, 4, 1600) for Conv1d
    real_tensor = torch.tensor(real_data, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(real_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ------------------------------------------------------------------
    # 2. Initialise models
    # ------------------------------------------------------------------
    G = Generator(latent_dim=latent_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        d_loss_epoch, g_loss_epoch = 0.0, 0.0

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            real_label = torch.ones(bs, 1, device=device)
            fake_label = torch.zeros(bs, 1, device=device)

            # --- Discriminator step ---
            z = torch.randn(bs, latent_dim, device=device)
            fake_batch = G(z).detach()

            d_real = D(real_batch)
            d_fake = D(fake_batch)
            loss_D = criterion(d_real, real_label) + criterion(d_fake, fake_label)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator step ---
            z = torch.randn(bs, latent_dim, device=device)
            fake_batch = G(z)
            d_fake = D(fake_batch)
            loss_G = criterion(d_fake, real_label)  # fool the discriminator

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            d_loss_epoch += loss_D.item()
            g_loss_epoch += loss_G.item()

        if epoch % 20 == 0 or epoch == 1:
            n_batches = len(loader)
            print(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"D_loss: {d_loss_epoch / n_batches:.4f}  "
                f"G_loss: {g_loss_epoch / n_batches:.4f}"
            )

    # ------------------------------------------------------------------
    # 4. Generate synthetic augmentation samples
    # ------------------------------------------------------------------
    n_synthetic = int(n_real * augment_ratio)
    print(f"\n[DCGAN] Generating {n_synthetic} synthetic windows ({augment_ratio*100:.0f}% of real)")

    G.eval()
    synthetic_windows = []
    with torch.no_grad():
        remaining = n_synthetic
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z).cpu().numpy()                       # (bs, 4, 1600)
            fake = np.transpose(fake, (0, 2, 1))            # (bs, 1600, 4)
            synthetic_windows.append(fake)
            remaining -= bs

    synthetic_data = np.concatenate(synthetic_windows, axis=0)

    # Assign approximate HR labels sampled from the real label distribution
    synthetic_labels = np.random.choice(real_labels, size=n_synthetic, replace=True)

    # ------------------------------------------------------------------
    # 5. Save synthetic dataset
    # ------------------------------------------------------------------
    out_dir = data_dir.parent / "synthetic_augmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "synthetic_data.npy", synthetic_data.astype(np.float32))
    np.save(out_dir / "synthetic_labels.npy", synthetic_labels.astype(np.float32))

    print(f"[DCGAN] Saved to {out_dir}/")
    print(f"  synthetic_data.npy  : {synthetic_data.shape}")
    print(f"  synthetic_labels.npy: {synthetic_labels.shape}")

    # Also save the generator checkpoint
    torch.save(G.state_dict(), out_dir / "generator.pt")
    print(f"  generator.pt        : saved")

    return synthetic_data, synthetic_labels


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN synthetic PPG generator")
    parser.add_argument("--data_dir", type=str, default="data/experiment_dataset_16s")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augment_ratio", type=float, default=0.15,
                        help="Fraction of synthetic data relative to real (0.15 = 15%%)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_dcgan(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment_ratio=args.augment_ratio,
        device=args.device,
    )
