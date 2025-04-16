import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        
        # Acoustic encoder
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(config.acoustic_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(config.hidden_dim, config.num_labels)
        
    def forward(self, text_inputs, acoustic_features, visual_features):
        # Text encoding
        text_outputs = self.text_encoder(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
        
        # Acoustic encoding
        acoustic_embeddings = self.acoustic_encoder(acoustic_features)
        
        # Visual encoding
        visual_embeddings = self.visual_encoder(visual_features)
        
        # Feature fusion
        fused_features = torch.cat([
            text_embeddings,
            acoustic_embeddings,
            visual_embeddings
        ], dim=1)
        
        fused_features = self.fusion_layer(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits 