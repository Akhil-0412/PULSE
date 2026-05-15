<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/HPC-Iridis-005A9C?style=for-the-badge" alt="HPC Iridis">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAkhil-0412%2FPULSE">
    <img src="https://vercel.com/button" alt="Deploy with Vercel" />
  </a>
</p>

<h1 align="center">🫀 PULSE</h1>
<h3 align="center"><b>P</b>PG-based <b>U</b>ncertainty-aware <b>L</b>earning for <b>S</b>ignal <b>E</b>stimation</h3>

<p align="center">
  <em>A vision-based deep learning pipeline for robust heart rate estimation from wearable PPG signals, with DCGAN synthetic data augmentation and conformal prediction for uncertainty quantification.</em>
</p>

---

## 📋 Overview

**PULSE** addresses critical challenges in wearable heart rate monitoring by building an end-to-end PyTorch pipeline that converts raw PPG signals into spectrogram representations, trains a CNN-BiLSTM-Attention hybrid on these 2D image inputs, and provides calibrated uncertainty estimates via Conformal Prediction.

### The Challenge
Traditional approaches fail on noisy, real-world data because they:
- ❌ Exclude "difficult" subjects from training
- ❌ Filter out low-quality signal windows
- ❌ Report inflated accuracy on clean-only subsets
- ❌ Provide no uncertainty estimate with predictions

### Our Solution
- ✅ Trains on **ALL subjects** including noisy, motion-corrupted samples
- ✅ Converts PPG signals to **spectrograms** — enabling the CNN to learn heart rate as a time-frequency pattern
- ✅ Augments training data by **15%** using a **DCGAN** synthetic data generator
- ✅ Provides **uncertainty bounds** via Conformal Prediction (90%+ coverage guarantee)

---

## 🏗️ Architecture

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PULSE PIPELINE                                      │
│                                                                                  │
│  Raw PPG Signal      Spectrogram         CNN Feature        BiLSTM Temporal      │
│  (4ch × 1600)    →   Representation  →   Extraction     →  Encoding         →    │
│                      (2D Image)          (Conv1d blocks)    (Bidirectional)      │
│                                                                                  │
│  ┌──────────┐       ┌──────────┐        ┌──────────┐      ┌──────────────┐       │
│  │ PPG      │       │ STFT     │        │ Conv1d   │      │ BiLSTM       │       │
│  │ + Accel  │  →    │ Log-Mag  │   →    │ 64→128   │  →   │ 256→128×2    │       │
│  │ 16s@100Hz│       │ Normalize│        │ →256     │      │ 2 layers     │       │
│  └──────────┘       └──────────┘        └──────────┘      └──────┬───────┘       │
│                                                                  │               │
│                   ┌──────────────────────────────────────────────┐│              │
│                   │          Temporal Attention                  ││              │
│                   │   Learns which time steps matter most       ◄┘│              │
│                   │   for heart rate estimation                   │              │
│                   └──────────────────┬───────────────────────────┘│              │
│                                      │                            │              │
│                               ┌──────▼──────┐                     │              │
│                               │  Regression │                     │              │
│                               │  Head       │                     │              │
│                               │  256→64→1   │                     │              │
│                               └──────┬──────┘                     │              │
│                                      │                            │              │
│                                      ▼                            │              │
│                              HR (BPM) ± Uncertainty               │              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Evolution

We iteratively improved our approach through three phases:

| Phase | Model | Architecture | MAE (BPM) | vs Baseline | Parameters |
|-------|-------|-------------|-----------|-------------|------------|
| 1 | TransPPG | Transformer Encoder | 11.11 | Baseline | ~3.2M |
| 2 | ResNet-1D | 4 Residual Blocks | 5.93 | **47% ↓** | ~2.0M |
| 3 | **AttentionCNNLSTM** | CNN + BiLSTM + Attention | **5.40** | **51% ↓** | ~1.5M |

### Key Results
- 🎯 **5.40 BPM** Mean Absolute Error (approaching clinical-grade accuracy)
- 📈 **90%+ Coverage** with Conformal Prediction uncertainty intervals
- 📊 **15% data augmentation** via DCGAN synthetic PPG generation
- ⚡ **Real-time inference** on consumer GPUs with mixed-precision (AMP)

---

## 🧬 Synthetic Data Generation (DCGAN)

A key contribution of this project is using a **Deep Convolutional Generative Adversarial Network** to synthesise realistic PPG windows, augmenting the training set by 15%.

### Why Synthetic Data?
- The PhysioNet dataset has only 22 subjects — limited diversity
- Motion-artifact-heavy segments are under-represented
- DCGAN generates plausible PPG morphologies the model hasn't seen, improving generalisation

### DCGAN Architecture

```
Generator:  z(128) → Linear → Reshape(512,25) → ConvT1d↑ → (4, 1600)
Discriminator:  (4, 1600) → Conv1d↓ (spectral norm) → Linear → real/fake
```

### Usage

```bash
# Train DCGAN and generate 15% augmentation
python -m src.augmentation.dcgan \
    --data_dir data/experiment_dataset_16s \
    --epochs 200 \
    --augment_ratio 0.15

# Output: data/synthetic_augmentation/
#   ├── synthetic_data.npy    (N, 1600, 4)
#   ├── synthetic_labels.npy  (N,)
#   └── generator.pt          (saved weights)
```

---

## 🖼️ Vision-Based Representation (Spectrograms)

Raw 1-D PPG signals are converted to **2-D spectrogram images** via Short-Time Fourier Transform (STFT). Heart rate manifests as a clear spectral peak in time-frequency space — this representation makes it significantly easier for convolutional filters to detect rhythmic patterns.

```bash
# Generate a sample spectrogram visualisation
python -m src.preprocessing.spectrograms
# Output: results/visualizations/spectrogram_S1_sample.png
```

```python
from src.preprocessing.spectrograms import ppg_to_spectrogram, PPGSpectrogramDataset

# Single window → spectrogram
spectrogram = ppg_to_spectrogram(ppg_window)  # (n_freq, n_time)

# Full PyTorch Dataset with on-the-fly spectrogram conversion
dataset = PPGSpectrogramDataset("data/experiment_dataset_16s", subject="S1")
spec_tensor, hr_label = dataset[0]  # (1, H, W), scalar
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Node.js 18+ (for web app — optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/Akhil-0412/PULSE.git
cd PULSE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Preprocess Data

```bash
# Download the PhysioNet PTT-PPG dataset first, then:
python -m src.preprocessing.preprocess_experiment
```

### 2. Generate Synthetic Augmentation (Optional)

```bash
python -m src.augmentation.dcgan --data_dir data/experiment_dataset_16s --epochs 200
```

### 3. Train Models

```bash
# Train Transformer baseline (Phase 1)
python src/training/train_transformer.py

# Train ResNet-1D baseline (Phase 2)
python src/training/train_resnet.py

# Train AttentionCNNLSTM — best model (Phase 3)
python src/training/train_hybrid.py
```

### 4. Run the Web App

```bash
# Start backend
cd webapp/backend
uvicorn main:app --reload --port 8000

# Start frontend (in new terminal)
cd frontend
npm install && npm run dev
```

Visit `http://localhost:3000` to explore the interactive dashboard.

---

## 📁 Project Structure

```
PULSE/
├── src/
│   ├── models/
│   │   ├── transformer.py           # Phase 1: TransPPG (baseline)
│   │   ├── resnet1d.py              # Phase 2: ResNet-1D (baseline)
│   │   └── attention_cnn_lstm.py    # Phase 3: CNN-BiLSTM-Attention (best)
│   ├── training/
│   │   ├── train_transformer.py     # LOSO training loop — Transformer
│   │   ├── train_resnet.py          # LOSO training loop — ResNet-1D
│   │   └── train_hybrid.py          # LOSO training loop — Hybrid (AMP + grad clip)
│   ├── preprocessing/
│   │   ├── preprocess_experiment.py # PhysioNet → windowed .npy files
│   │   ├── preprocess_real.py       # Real-time signal preprocessing
│   │   └── spectrograms.py          # PPG → 2D spectrogram conversion (STFT)
│   ├── augmentation/
│   │   └── dcgan.py                 # DCGAN synthetic PPG data generator
│   └── utils/
│       ├── analyze_ranking.py       # Per-subject MAE ranking analysis
│       ├── check_results.py         # Results validation
│       ├── verify_loso.py           # Leave-One-Subject-Out verification
│       └── visualize.py             # Plotting utilities
├── results/
│   ├── hybrid/                      # Saved model results (.npy)
│   ├── resnet1d/
│   └── visualizations/              # Generated plots and spectrograms
├── app/                             # Next.js App Router (web dashboard)
├── webapp/backend/                  # FastAPI inference server
├── requirements.txt                 # Pinned Python dependencies
└── README.md
```

---

## 🧠 Model Deep Dive: AttentionCNNLSTM

```
Input: 4 channels × 1600 samples (16 seconds @ 100Hz)
                    │
    ┌───────────────▼───────────────┐
    │     CNN Feature Extraction    │
    │   Conv1d(4→64→128→256)        │
    │   BatchNorm + ReLU + MaxPool  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │    Bi-LSTM Temporal Encoder   │
    │   LSTM(256→128, bidir, 2L)    │
    │   + gradient clipping (1.0)   │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │      Temporal Attention       │
    │   Linear(256→64) + Tanh       │
    │   Softmax weighted sum        │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │     Regression + Dropout      │
    │   Dense(256→64→1)             │
    └───────────────┬───────────────┘
                    │
                    ▼
            Heart Rate (BPM)
            ± Uncertainty (via Conformal Prediction)
```

### Training Details
- **Optimiser**: AdamW (lr=1e-4, weight_decay=1e-2)
- **Scheduler**: Cosine Annealing over 50 epochs
- **Mixed Precision**: Automatic Mixed Precision (AMP) with GradScaler
- **Validation**: Leave-One-Subject-Out (LOSO) cross-validation across all 22 subjects
- **Uncertainty**: Split Conformal Prediction with α=0.1 (90% target coverage)

---

## 📚 Dataset

Uses the **PhysioNet Pulse Transit Time PPG Dataset v1.1.0**:
- 22 healthy subjects performing sit / walk / run activities
- 4-channel PPG + 3-axis accelerometry
- Ground truth ECG-derived heart rate
- 16-second sliding windows with 75% overlap (step = 4s)

> ⚠️ Dataset files not included due to licensing. Download from [PhysioNet](https://physionet.org/).

---

## 🙏 Acknowledgments

- **Dataset**: PhysioNet PTT-PPG Dataset v1.1.0
- **Frameworks**: PyTorch, FastAPI, Next.js, Recharts
- **Compute**: University of Southampton HPC Iridis Multi-Cluster (GPU nodes)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>PULSE</b> — PPG-based Uncertainty-aware Learning for Signal Estimation<br>
  <em>Built with PyTorch · Trained on HPC Iridis · University of Southampton MSc AI Dissertation</em>
</p>
