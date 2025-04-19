import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=128, num_layers=2):
        super(AudioEncoder, self).__init__()
        
        # 简化的音频编码器，适用于一维特征
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, 1)
        x = x.squeeze(1)  # (batch_size, 1)
        return self.encoder(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128):
        super(TextEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.transformer.config.hidden_size, hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to final dimension
        output = self.projection(cls_output)
        return output

class DecisionFusionModel(nn.Module):
    def __init__(self, audio_input_dim=74, hidden_dim=128, num_classes=3):
        super(DecisionFusionModel, self).__init__()
        
        # Initialize encoders
        self.audio_encoder = AudioEncoder(
            input_dim=audio_input_dim,
            hidden_dim=hidden_dim
        )
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)
        
        # Individual classifiers
        self.audio_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Fusion weights (learnable)
        self.audio_weight = nn.Parameter(torch.tensor(0.5))
        self.text_weight = nn.Parameter(torch.tensor(0.5))
        
        # Softmax for fusion weights
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, audio_input, text_input_ids, text_attention_mask):
        # Process audio
        audio_features = self.audio_encoder(audio_input)
        audio_logits = self.audio_classifier(audio_features)
        
        # Process text
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        text_logits = self.text_classifier(text_features)
        
        # Normalize fusion weights
        weights = self.softmax(torch.stack([self.audio_weight, self.text_weight]))
        
        # Weighted fusion of logits
        fused_logits = weights[0] * audio_logits + weights[1] * text_logits
        
        return {
            'fused_logits': fused_logits,
            'audio_logits': audio_logits,
            'text_logits': text_logits,
            'weights': weights
        }
    
    def get_fusion_weights(self):
        """Get the current fusion weights"""
        weights = self.softmax(torch.stack([self.audio_weight, self.text_weight]))
        return {
            'audio_weight': weights[0].item(),
            'text_weight': weights[1].item()
        } 