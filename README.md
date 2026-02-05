<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Next.js-14+-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">🫀 PULSE</h1>
<h3 align="center"><b>P</b>PG-based <b>U</b>ncertainty-aware <b>L</b>earning for <b>S</b>ignal <b>E</b>stimation</h3>

<p align="center">
  <em>A robust deep learning system for heart rate estimation from PPG signals with conformal prediction for uncertainty quantification.</em>
</p>

---

## 📋 Overview

**PULSE** addresses critical challenges in wearable heart rate monitoring by developing robust deep learning models that work reliably on noisy, real-world PPG (Photoplethysmography) signals.

### The Challenge
Traditional approaches fail on noisy data because they:
- ❌ Exclude "difficult" subjects from training
- ❌ Filter out low-quality signal windows
- ❌ Report inflated accuracy metrics

### Our Solution
We developed an end-to-end learning approach that:
- ✅ Trains on **ALL subjects** including noisy data
- ✅ Uses **16-second windows** for robust rhythm detection
- ✅ Provides **uncertainty estimates** via Conformal Prediction

---

## 🏗️ Architecture Evolution

We iteratively improved our approach through three phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARCHITECTURE EVOLUTION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Transformer        Phase 2: ResNet-1D       Phase 3: Hybrid       │
│  ┌─────────────────┐         ┌─────────────────┐      ┌─────────────────┐   │
│  │  Self-Attention │         │   Conv Blocks   │      │   CNN + LSTM    │   │
│  │    Encoder      │   →     │   + Residual    │  →   │  + Attention    │   │
│  │   (11.11 BPM)   │         │   (5.93 BPM)    │      │   (5.40 BPM)    │   │
│  └─────────────────┘         └─────────────────┘      └─────────────────┘   │
│                                                                             │
│  Issue: Overfitting          47% Improvement!         BEST MODEL - Custom   │
│  to motion noise             via signal filtering     architecture design   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Performance Comparison

| Model | Architecture | MAE (BPM) | Improvement | Parameters |
|-------|--------------|-----------|-------------|------------|
| TransPPG | Transformer Encoder | 11.11 | Baseline | ~3.2M |
| ResNet-1D | 4 Residual Blocks | 5.93 | **47% ↓** | ~2.0M |
| **AttentionCNNLSTM** | CNN + Bi-LSTM + Attention | **5.40** | **51% ↓** | ~1.5M |

### Key Features
- 🎯 **5.40 BPM** Mean Absolute Error (approaching clinical-grade)
- 📈 **90%+ Coverage** with Conformal Prediction intervals
- ⚡ **Real-time Inference** on consumer GPUs

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Node.js 18+ (for web app)

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

### Training Models

```bash
# Train Transformer (Phase 1)
python src/training/train_transformer.py

# Train ResNet-1D (Phase 2)
python src/training/train_resnet.py

# Train AttentionCNNLSTM (Phase 3 - Best)
python src/training/train_hybrid.py
```

### Running the Web App

```bash
# Start backend
cd webapp/backend
uvicorn main:app --reload --port 8000

# Start frontend (in new terminal)
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to explore the interactive dashboard.

---

## 📁 Project Structure

```
PULSE/
├── src/
│   ├── models/
│   │   ├── transformer.py        # Phase 1: TransPPG
│   │   ├── resnet1d.py           # Phase 2: ResNet-1D
│   │   └── attention_cnn_lstm.py # Phase 3: Custom Hybrid
│   ├── training/
│   │   ├── train_transformer.py
│   │   ├── train_resnet.py
│   │   └── train_hybrid.py
│   ├── preprocessing/
│   └── utils/
├── results/
│   ├── resnet1d/
│   ├── hybrid/
│   └── visualizations/
├── frontend/                     # Next.js dashboard
├── webapp/backend/               # FastAPI server
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture: AttentionCNNLSTM

```
Input: 4 channels × 1600 samples (16 seconds @ 100Hz)
                    │
    ┌───────────────▼───────────────┐
    │     CNN Feature Extraction    │
    │   Conv1d(4→64→128→256)        │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │      Bi-LSTM Temporal         │
    │    LSTM(256→128, bidir)       │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │    Attention + Regression     │
    │   Dense(256→64→1)             │
    └───────────────┬───────────────┘
                    │
                    ▼
            Heart Rate (BPM)
            + Uncertainty (±BPM)
```

---

## 📚 Dataset

Uses the **PhysioNet Pulse Transit Time PPG Dataset v1.1.0**:
- 22 healthy subjects
- 4-channel PPG + 3-axis accelerometry
- Ground truth ECG-derived heart rate

> ⚠️ Dataset files not included. Download from [PhysioNet](https://physionet.org/).

---

## 🙏 Acknowledgments

- **Dataset**: PhysioNet PTT-PPG Dataset
- **Frameworks**: PyTorch, FastAPI, Next.js, Recharts

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>PULSE</b> — PPG-based Uncertainty-aware Learning for Signal Estimation
</p>
