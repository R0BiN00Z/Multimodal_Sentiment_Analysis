import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class FeatureAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(FeatureAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.dropout(out)
        
        # Add & Norm
        out = self.layer_norm(x + out)
        return out

class CrossModalAttention(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projections for audio features
        self.audio_query = nn.Linear(audio_dim, hidden_dim)
        self.audio_key = nn.Linear(audio_dim, hidden_dim)
        self.audio_value = nn.Linear(audio_dim, hidden_dim)
        
        # Projections for text features
        self.text_query = nn.Linear(text_dim, hidden_dim)
        self.text_key = nn.Linear(text_dim, hidden_dim)
        self.text_value = nn.Linear(text_dim, hidden_dim)
        
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio_features, text_features):
        batch_size = audio_features.size(0)
        seq_len = text_features.size(1)
        
        # Project audio features
        audio_q = self.audio_query(audio_features)
        audio_k = self.audio_key(audio_features)
        audio_v = self.audio_value(audio_features)
        
        # Project text features
        text_q = self.text_query(text_features)
        text_k = self.text_key(text_features)
        text_v = self.text_value(text_features)
        
        # Reshape for attention
        audio_q = audio_q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        audio_k = audio_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        audio_v = audio_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        text_q = text_q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        text_k = text_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        text_v = text_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-modal attention
        audio_scores = torch.matmul(audio_q, text_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        text_scores = torch.matmul(text_q, audio_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        audio_attn = F.softmax(audio_scores, dim=-1)
        text_attn = F.softmax(text_scores, dim=-1)
        
        audio_attn = self.dropout(audio_attn)
        text_attn = self.dropout(text_attn)
        
        # Apply attention
        audio_out = torch.matmul(audio_attn, text_v)
        text_out = torch.matmul(text_attn, audio_v)
        
        # Reshape back
        audio_out = audio_out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        text_out = text_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Average pooling over sequence dimension for text
        text_out = text_out.mean(dim=1, keepdim=True)
        
        # Concatenate and project
        out = torch.cat([audio_out, text_out], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class AdvancedFusionModel(nn.Module):
    def __init__(self, audio_input_dim=1, text_hidden_dim=768, hidden_dim=512, num_classes=5):
        super(AdvancedFusionModel, self).__init__()
        
        # Audio feature processing
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_input_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Text feature processing
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature attention
        self.audio_attention = FeatureAttention(hidden_dim, hidden_dim)
        self.text_attention = FeatureAttention(hidden_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, audio_input, text_input_ids, text_attention_mask):
        # Process audio features
        # Ensure audio input has shape (batch_size, 1)
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(1)
        
        # Project audio to hidden dimension
        audio_features = self.audio_proj(audio_input)
        # Add sequence dimension for attention
        audio_features = audio_features.unsqueeze(1)
        audio_features = self.audio_attention(audio_features)
        
        # Process text features
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state
        text_features = self.text_proj(text_features)
        text_features = self.text_attention(text_features)
        
        # Cross-modal attention
        fused_features = self.cross_attention(audio_features, text_features)
        
        # Ensure fused features have correct shape
        if fused_features.dim() == 3:
            fused_features = fused_features.squeeze(1)
        
        # Apply fusion layers
        fused_features = self.fusion(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        return logits 