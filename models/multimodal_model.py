import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=128, num_layers=2):
        super(AudioEncoder, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, 512)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Project to final dimension
        output = self.projection(last_hidden)
        return output

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

class MultimodalModel(nn.Module):
    def __init__(self, audio_input_dim=74, hidden_dim=128, num_classes=3):
        super(MultimodalModel, self).__init__()
        
        # Initialize encoders
        self.audio_encoder = AudioEncoder(
            input_dim=audio_input_dim,
            hidden_dim=hidden_dim
        )
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, audio_input, text_input_ids, text_attention_mask):
        # Process audio
        audio_features = self.audio_encoder(audio_input)
        
        # Process text
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        # Concatenate features
        combined_features = torch.cat([audio_features, text_features], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        return output 