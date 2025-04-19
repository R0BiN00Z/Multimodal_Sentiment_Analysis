import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(TextModel, self).__init__()
        
        # 使用预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 文本特征处理
        self.text_encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),  # BERT输出维度是768
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS]标记的输出作为文本表示
        text_features = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 文本特征处理
        text_features = self.text_encoder(text_features)  # [batch_size, hidden_dim]
        
        # 分类
        logits = self.classifier(text_features)
        
        return logits 