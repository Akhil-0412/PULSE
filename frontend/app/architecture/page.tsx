'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';

// Architecture diagrams as ASCII art (will be styled with code blocks)
const TRANSFORMER_ARCH = `
┌─────────────────────────────────────────────────────────┐
│              Input: PPG + Accelerometer                 │
│              (4 channels × 4000 samples @ 400Hz)        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Token Embedding                        │
│                   Conv1D + Linear                        │
│                   Projects to d_model=128               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Positional Encoding                        │
│               Sinusoidal (Learnable)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────┴────────────────┐
          │      Transformer Encoder        │
          │      (4 Layers)                 │
          │                                 │
          │  ┌───────────────────────────┐  │
          │  │  Multi-Head Attention     │  │
          │  │  (8 Heads)                │  │
          │  └───────────────────────────┘  │
          │              │                  │
          │              ▼                  │
          │  ┌───────────────────────────┐  │
          │  │  Add & Norm               │  │
          │  └───────────────────────────┘  │
          │              │                  │
          │              ▼                  │
          │  ┌───────────────────────────┐  │
          │  │  Feed Forward (512)       │  │
          │  └───────────────────────────┘  │
          │              │                  │
          │              ▼                  │
          │  ┌───────────────────────────┐  │
          │  │  Add & Norm               │  │
          │  └───────────────────────────┘  │
          └────────────────┬────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Global Average Pooling                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Output: Heart Rate (BPM)                 │
│                Linear(128 → 1)                          │
└─────────────────────────────────────────────────────────┘
`;

const RESNET_ARCH = `
┌─────────────────────────────────────────────────────────┐
│              Input: PPG + Accelerometer                 │
│              (4 channels × 1600 samples @ 100Hz)        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Initial Conv Block                     │
│         Conv1D(4→64, k=15, s=2) + BN + ReLU            │
│                 MaxPool(k=3, s=2)                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               ResNet Block 1 (×2)                       │
│        ┌─────────────────────────────────┐             │
│        │  Conv1D(64→64, k=3) + BN + ReLU │             │
│        │  Conv1D(64→64, k=3) + BN        │             │
│        │         + Skip Connection       │             │
│        └─────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               ResNet Block 2 (×2)                       │
│         Conv1D(64→128, k=3) + Downsample               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               ResNet Block 3 (×2)                       │
│         Conv1D(128→256, k=3) + Downsample              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               ResNet Block 4 (×2)                       │
│         Conv1D(256→512, k=3) + Downsample              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│            Adaptive Average Pooling (1)                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Output: Heart Rate (BPM)                 │
│                Linear(512 → 1)                          │
└─────────────────────────────────────────────────────────┘
`;

const CNNLSTM_ARCH = `
┌─────────────────────────────────────────────────────────┐
│              Input: PPG + Accelerometer                 │
│              (4 channels × 1600 samples @ 100Hz)        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Conv Block 1                           │
│       Conv1D(4→64, k=15, s=2) + BN + ReLU              │
│              MaxPool(k=3, s=2)                          │
│           Output: (64, 400)                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Conv Block 2                           │
│       Conv1D(64→128, k=7, s=2) + BN + ReLU             │
│              MaxPool(k=3, s=2)                          │
│           Output: (128, 100)                            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Conv Block 3                           │
│       Conv1D(128→256, k=5, s=2) + BN + ReLU            │
│           Output: (256, 50)                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Reshape                               │
│           (256, 50) → (50, 256)                        │
│           Sequence length = 50                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Bidirectional LSTM                       │
│           LSTM(256→128, layers=2)                      │
│              Dropout = 0.3                              │
│           Output: (50, 256)                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Temporal Attention                       │
│       Learn weights for each timestep                   │
│       Softmax → Weighted Sum                            │
│           Output: (256)                                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Regression Head                           │
│       Dropout(0.3) → Dense(256→64) → ReLU              │
│              Dense(64→1)                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Output: Heart Rate (BPM)                 │
└─────────────────────────────────────────────────────────┘
`;

const models = [
    {
        id: 'transformer',
        name: 'TransPPG (Transformer)',
        description: 'Self-attention based architecture for capturing long-range temporal dependencies in PPG signals.',
        params: '~3.2M',
        mae: '11.10',
        status: 'Baseline',
        architecture: TRANSFORMER_ARCH,
        pros: ['Global context via attention', 'Parallel processing', 'Handles variable lengths'],
        cons: ['High compute cost', 'Needs lots of data', 'Struggled with noisy subjects'],
        color: 'from-purple-500 to-violet-600'
    },
    {
        id: 'resnet',
        name: 'ResNet-1D',
        description: 'Deep residual network adapted for 1D time-series with skip connections and multi-scale feature extraction.',
        params: '~2.0M',
        mae: '5.93',
        status: 'Improved',
        architecture: RESNET_ARCH,
        pros: ['Skip connections prevent gradient vanishing', 'Efficient training', 'Strong local feature extraction'],
        cons: ['Limited long-range modeling', 'Global pooling loses temporal info'],
        color: 'from-cyan-500 to-teal-600'
    },
    {
        id: 'cnnlstm',
        name: 'AttentionCNNLSTM',
        description: 'Hybrid architecture combining CNN feature extraction with bidirectional LSTM temporal modeling and attention.',
        params: '~1.05M',
        mae: '5.40',
        status: 'Best',
        architecture: CNNLSTM_ARCH,
        pros: ['CNN extracts local features', 'LSTM captures temporal dependencies', 'Attention focuses on important timesteps', 'Smallest model'],
        cons: ['Sequential LSTM slower than CNN', 'More hyperparameters to tune'],
        color: 'from-rose-500 to-pink-600'
    }
];

export default function ArchitecturePage() {
    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
            {/* Header */}
            <header className="border-b border-slate-800/50 backdrop-blur-xl bg-slate-950/80 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 group">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-teal-500 flex items-center justify-center">
                            <span className="text-white text-lg">❤️</span>
                        </div>
                        <span className="font-bold text-xl bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                            PPG Analysis
                        </span>
                    </Link>
                    <nav className="flex items-center gap-6">
                        <Link href="/dashboard" className="text-slate-400 hover:text-white transition-colors">
                            Dashboard
                        </Link>
                        <Link href="/architecture" className="text-rose-400 font-medium">
                            Architecture
                        </Link>
                    </nav>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-6 py-12">
                {/* Title Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="text-center mb-12"
                >
                    <h1 className="text-4xl font-bold text-white mb-4">
                        Model Architectures
                    </h1>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                        Compare the three architectures explored for PPG-based heart rate estimation.
                        From Transformer to ResNet to our custom CNN-LSTM hybrid.
                    </p>
                </motion.div>

                {/* Comparison Table */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="mb-16"
                >
                    <div className="bg-slate-900/50 rounded-2xl border border-slate-800 overflow-hidden">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-slate-800">
                                    <th className="text-left px-6 py-4 text-slate-400 font-medium">Model</th>
                                    <th className="text-center px-6 py-4 text-slate-400 font-medium">Parameters</th>
                                    <th className="text-center px-6 py-4 text-slate-400 font-medium">Avg MAE (BPM)</th>
                                    <th className="text-center px-6 py-4 text-slate-400 font-medium">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {models.map((model, idx) => (
                                    <tr key={model.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${model.color}`} />
                                                <span className="text-white font-medium">{model.name}</span>
                                            </div>
                                        </td>
                                        <td className="text-center px-6 py-4 text-slate-300 font-mono">{model.params}</td>
                                        <td className="text-center px-6 py-4">
                                            <span className={`font-bold ${model.status === 'Best' ? 'text-green-400' : model.status === 'Improved' ? 'text-cyan-400' : 'text-slate-400'}`}>
                                                {model.mae}
                                            </span>
                                        </td>
                                        <td className="text-center px-6 py-4">
                                            <span className={`px-3 py-1 rounded-full text-xs font-medium ${model.status === 'Best' ? 'bg-green-500/20 text-green-400' :
                                                    model.status === 'Improved' ? 'bg-cyan-500/20 text-cyan-400' :
                                                        'bg-slate-700 text-slate-400'
                                                }`}>
                                                {model.status}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>

                {/* Architecture Cards */}
                <div className="space-y-12">
                    {models.map((model, idx) => (
                        <motion.div
                            key={model.id}
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.2 + idx * 0.1 }}
                            className="bg-slate-900/50 rounded-2xl border border-slate-800 overflow-hidden"
                        >
                            {/* Card Header */}
                            <div className={`bg-gradient-to-r ${model.color} p-6`}>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h2 className="text-2xl font-bold text-white">{model.name}</h2>
                                        <p className="text-white/80 mt-1">{model.description}</p>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-3xl font-bold text-white">{model.mae}</div>
                                        <div className="text-white/60 text-sm">BPM MAE</div>
                                    </div>
                                </div>
                            </div>

                            {/* Card Body */}
                            <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Architecture Diagram */}
                                <div className="lg:col-span-2">
                                    <h3 className="text-slate-400 font-medium mb-3">Architecture Diagram</h3>
                                    <pre className="bg-slate-950 rounded-xl p-4 overflow-x-auto text-xs text-slate-300 font-mono border border-slate-800">
                                        {model.architecture}
                                    </pre>
                                </div>

                                {/* Pros & Cons */}
                                <div className="space-y-6">
                                    <div>
                                        <h3 className="text-slate-400 font-medium mb-3">✅ Advantages</h3>
                                        <ul className="space-y-2">
                                            {model.pros.map((pro, i) => (
                                                <li key={i} className="flex items-start gap-2 text-sm text-green-400">
                                                    <span className="mt-1">•</span>
                                                    <span>{pro}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    <div>
                                        <h3 className="text-slate-400 font-medium mb-3">⚠️ Limitations</h3>
                                        <ul className="space-y-2">
                                            {model.cons.map((con, i) => (
                                                <li key={i} className="flex items-start gap-2 text-sm text-yellow-400">
                                                    <span className="mt-1">•</span>
                                                    <span>{con}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    <div className="pt-4 border-t border-slate-800">
                                        <div className="flex items-center justify-between text-sm">
                                            <span className="text-slate-500">Parameters</span>
                                            <span className="text-white font-mono">{model.params}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* Back to Dashboard */}
                <div className="mt-12 text-center">
                    <Link
                        href="/dashboard"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-rose-500 to-teal-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity"
                    >
                        ← Back to Dashboard
                    </Link>
                </div>
            </main>
        </div>
    );
}
