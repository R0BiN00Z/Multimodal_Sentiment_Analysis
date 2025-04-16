import torch
import torch.nn as nn

class BimodalSentimentModel(nn.Module):
    def __init__(self, text_input_size, audio_input_size, hidden_size, num_classes, dropout=0.1):
        super(BimodalSentimentModel, self).__init__()
        
        # Text feature processing
        self.text_fc = nn.Linear(text_input_size, hidden_size)
        self.text_bn = nn.BatchNorm1d(hidden_size)
        
        # Audio feature processing
        self.audio_fc = nn.Linear(audio_input_size, hidden_size)
        self.audio_bn = nn.BatchNorm1d(hidden_size)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_bn = nn.BatchNorm1d(hidden_size)
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, text, audio):
        # Process text features
        text_features = self.text_fc(text)
        text_features = self.text_bn(text_features)
        text_features = self.relu(text_features)
        text_features = self.dropout(text_features)
        
        # Process audio features
        audio_features = self.audio_fc(audio)
        audio_features = self.audio_bn(audio_features)
        audio_features = self.relu(audio_features)
        audio_features = self.dropout(audio_features)
        
        # Concatenate features
        combined = torch.cat([text_features, audio_features], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        fused = self.fusion_bn(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        
        # Classification
        output = self.classifier(fused)
        
        return output 