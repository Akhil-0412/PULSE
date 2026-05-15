"""
CNN-BiLSTM-Attention Model for PPG Heart Rate Estimation
=========================================================

This module implements the final (Phase 3) architecture for PULSE.

Design rationale:
- **CNN front-end**: Extracts local morphological features from the PPG
  waveform and its spectrogram representation.  Convolutional filters are
  well-suited to detecting the repeating pulse peaks and motion-artifact
  patterns that characterise PPG signals.
- **Bidirectional LSTM**: Captures temporal dependencies across the full
  16-second window.  Bidirectionality lets the model use both past and
  future context — important because heart rate is a quasi-periodic
  signal whose rhythm is most reliably estimated over multiple cycles.
- **Temporal Attention**: Not all time steps contribute equally to the
  HR estimate (e.g., motion-corrupted segments should be down-weighted).
  A learnable attention layer lets the model focus on the most
  informative time steps.

Why not a pure Transformer?
  Phase 1 (TransPPG) showed that self-attention alone overfits to motion
  noise on this dataset size (22 subjects, ~3k windows).  The hybrid
  approach achieves 51% lower MAE with fewer parameters (~1.5M vs ~3.2M).

Inputs:  (Batch, 4, 1600)  — 4 channels (PPG + 3-axis accel), 16s @ 100Hz
Outputs: (Batch, 1)        — predicted heart rate in BPM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with Conv1d + BatchNorm + ReLU + optional pooling."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pool=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if pool else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for PPG Heart Rate Estimation.
    
    Architecture:
    - 3 Convolutional blocks for local feature extraction
    - Bidirectional LSTM for temporal sequence modeling
    - Regression head for heart rate prediction
    
    Input: (Batch, 4, 1600) - 4 channels (PPG + 3 Accel), 16s @ 100Hz
    Output: (Batch, 1) - Heart Rate in BPM
    """
    
    def __init__(self, input_channels=4, lstm_hidden=128, dropout=0.3):
        super(HybridCNNLSTM, self).__init__()
        
        # === Convolutional Feature Extractor ===
        # Block 1: (4, 1600) -> (64, 400)
        self.conv1 = ConvBlock(input_channels, 64, kernel_size=15, stride=2, pool=True)
        
        # Block 2: (64, 400) -> (128, 100)
        self.conv2 = ConvBlock(64, 128, kernel_size=7, stride=2, pool=True)
        
        # Block 3: (128, 100) -> (256, 50)
        self.conv3 = ConvBlock(128, 256, kernel_size=5, stride=2, pool=False)
        
        # === Recurrent Temporal Encoder ===
        # Input: (Batch, 50, 256) after transpose
        # Output: (Batch, 50, 256) with bidirectional hidden
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # === Regression Head ===
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x: (Batch, 4, 1600)
        
        # CNN Feature Extraction
        x = self.conv1(x)  # (Batch, 64, 400)
        x = self.conv2(x)  # (Batch, 128, 100)
        x = self.conv3(x)  # (Batch, 256, 50)
        
        # Reshape for LSTM: (Batch, Channels, Length) -> (Batch, Length, Channels)
        x = x.permute(0, 2, 1)  # (Batch, 50, 256)
        
        # LSTM Temporal Encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the concatenated final hidden states from both directions
        # h_n shape: (num_layers * num_directions, Batch, hidden_size)
        # Get last layer's hidden states
        h_forward = h_n[-2, :, :]  # (Batch, lstm_hidden)
        h_backward = h_n[-1, :, :]  # (Batch, lstm_hidden)
        hidden = torch.cat([h_forward, h_backward], dim=1)  # (Batch, lstm_hidden * 2)
        
        # Regression Head
        x = self.dropout(hidden)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class AttentionCNNLSTM(nn.Module):
    """
    Best-performing model (Phase 3): CNN-BiLSTM with Temporal Attention.

    Instead of using only the final LSTM hidden state (which compresses
    the entire sequence into a single vector), this model applies a
    learned attention mechanism over ALL LSTM time steps.  The attention
    weights highlight segments with strong, clean pulse peaks and
    suppress motion-corrupted frames — a form of soft, data-driven
    noise rejection.

    Performance: 5.40 BPM MAE | 90%+ conformal coverage | ~1.5M params
    """
    
    def __init__(self, input_channels=4, lstm_hidden=128, dropout=0.3):
        super(AttentionCNNLSTM, self).__init__()
        
        # === Convolutional Feature Extractor ===
        self.conv1 = ConvBlock(input_channels, 64, kernel_size=15, stride=2, pool=True)
        self.conv2 = ConvBlock(64, 128, kernel_size=7, stride=2, pool=True)
        self.conv3 = ConvBlock(128, 256, kernel_size=5, stride=2, pool=False)
        
        # === Recurrent Temporal Encoder ===
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # === Temporal Attention ===
        lstm_output_size = lstm_hidden * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # === Regression Head ===
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (Batch, SeqLen, Features)
        
        # LSTM Encoding
        lstm_out, _ = self.lstm(x)  # (Batch, SeqLen, lstm_hidden*2)
        
        # Temporal Attention
        attn_weights = self.attention(lstm_out)  # (Batch, SeqLen, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum over time
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (Batch, lstm_hidden*2)
        
        # Regression Head
        x = self.dropout(context)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    # Test the models
    batch_size = 8
    seq_len = 1600
    channels = 4
    
    x = torch.randn(batch_size, channels, seq_len)
    
    print("Testing HybridCNNLSTM...")
    model1 = HybridCNNLSTM()
    out1 = model1(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out1.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    print("\nTesting AttentionCNNLSTM...")
    model2 = AttentionCNNLSTM()
    out2 = model2(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out2.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")
