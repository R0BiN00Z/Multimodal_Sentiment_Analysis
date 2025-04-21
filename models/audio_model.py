import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioNet(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=128, num_classes=5):
        super(AudioNet, self).__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Simplified CNN layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Simplified LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        # Input shape: [batch_size, input_dim, seq_len]
        
        # Input normalization
        x = x.transpose(1, 2)  # [batch_size, seq_len, input_dim]
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # LSTM processing
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_dim * 2]
        
        # Classification
        x = self.classifier(pooled)
        
        return x 