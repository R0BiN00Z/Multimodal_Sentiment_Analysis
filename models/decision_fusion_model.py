import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=256):
        super(AudioEncoder, self).__init__()
        
        # Input normalization and projection
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Simple CNN layers
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # Input shape: [batch_size, input_dim, seq_len]
        batch_size = x.size(0)
        
        # Normalize and project
        x = x.transpose(1, 2)  # [batch_size, seq_len, input_dim]
        x = self.layer_norm(x)
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        return x

class TextEncoder(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256):
        super(TextEncoder, self).__init__()
        
        # Input normalization and projection
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Simple attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Normalize and project
        x = self.layer_norm(x)
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Simple attention
        attn_weights = F.softmax(self.attention(x), dim=1)  # [batch_size, seq_len, 1]
        x = torch.bmm(x.transpose(1, 2), attn_weights).squeeze(-1)  # [batch_size, hidden_dim]
        
        return x

class DecisionFusionModel(nn.Module):
    def __init__(self, audio_input_dim=74, text_input_dim=300, hidden_dim=256, num_classes=5):
        super(DecisionFusionModel, self).__init__()
        
        # Initialize encoders
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim)
        self.text_encoder = TextEncoder(text_input_dim, hidden_dim)
        
        # Classifiers
        self.audio_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize fusion weights
        self.audio_weight = nn.Parameter(torch.tensor(0.5))
        self.text_weight = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, audio_input, text_input):
        # Process audio
        audio_features = self.audio_encoder(audio_input)
        audio_logits = self.audio_classifier(audio_features)
        
        # Process text
        text_features = self.text_encoder(text_input)
        text_logits = self.text_classifier(text_features)
        
        # Weighted fusion with temperature scaling
        weights = F.softmax(torch.stack([self.audio_weight, self.text_weight]) / 0.5, dim=0)
        fused_logits = weights[0] * audio_logits + weights[1] * text_logits
        
        return fused_logits, text_logits, audio_logits
    
    def get_fusion_weights(self):
        weights = F.softmax(torch.stack([self.audio_weight, self.text_weight]) / 0.5, dim=0)
        return {
            'audio_weight': weights[0].item(),
            'text_weight': weights[1].item()
        } 