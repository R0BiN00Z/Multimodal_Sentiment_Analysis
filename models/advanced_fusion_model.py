import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection with layer normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.transformer(x)
        return x.mean(dim=1)  # Global average pooling

class TextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Initial projection with layer normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.transformer(x)
        return x.mean(dim=1)  # Global average pooling

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key_value):
        # query, key_value: [batch_size, seq_len, hidden_dim]
        residual = query
        attn_output, _ = self.multihead_attn(
            query=self.layer_norm1(query),
            key=self.layer_norm2(key_value),
            value=self.layer_norm2(key_value)
        )
        return residual + self.dropout(attn_output)

class AdvancedFusionModel(nn.Module):
    def __init__(self, audio_input_dim=74, text_input_dim=300, hidden_dim=256, num_classes=5):
        super().__init__()
        
        # Feature encoders
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim)
        self.text_encoder = TextEncoder(text_input_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, audio_input, text_input):
        # Extract features
        audio_features = self.audio_encoder(audio_input)
        text_features = self.text_encoder(text_input)
        
        # Cross-modal attention
        attended_features = self.cross_attention(
            audio_features.unsqueeze(1),
            text_features.unsqueeze(1)
        ).squeeze(1)
        
        # Concatenate features
        fused_features = torch.cat([audio_features, attended_features], dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits 