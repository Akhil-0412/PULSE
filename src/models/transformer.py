
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransPPG(nn.Module):
    def __init__(self, input_channels=4, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransPPG, self).__init__()
        
        # 1. Embedding Layer (CNN to tokenize the signal)
        # Input shape: (Batch, Channels, Length) -> (Batch, 4, 4000)
        # We want to reduce length to make it manageable for Transformer (e.g., 4000 -> 250 tokens)
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Regression Head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Heart Rate
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        
        # Embedding
        x = self.embedding(x) # (Batch, d_model, Length/16)
        
        # Permute for Transformer (Seq_Len, Batch, d_model)
        x = x.permute(2, 0, 1)
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Pooling (Back to Batch first for pooling)
        x = x.permute(1, 2, 0) # (Batch, d_model, Seq_Len)
        x = self.avg_pool(x).squeeze(-1) # (Batch, d_model)
        
        # Regression
        hr = self.regressor(x)
        
        return hr
