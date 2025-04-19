import torch
import torch.nn as nn

class AudioModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AudioModel, self).__init__()
        
        # 音频特征处理
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, audio):
        # 音频特征处理
        audio_features = self.audio_encoder(audio.squeeze(1))  # [batch_size, hidden_dim, 1]
        audio_features = audio_features.squeeze(-1)  # [batch_size, hidden_dim]
        
        # 分类
        logits = self.classifier(audio_features)
        
        return logits 