import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.decision_fusion_model import DecisionFusionModel

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='train'):
        self.text_data = np.load(text_path)
        self.audio_data = np.load(audio_path)
        self.labels = np.load(label_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 确保数据对齐
        assert len(self.text_data) == len(self.audio_data) == len(self.labels), \
            f"Data lengths mismatch: text={len(self.text_data)}, audio={len(self.audio_data)}, labels={len(self.labels)}"
        
        # 打印数据集信息
        print(f"\n{split} 集信息:")
        print(f"样本数量: {len(self.labels)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(self.labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        # 获取文本和音频特征
        text = self.text_data[idx]
        audio = self.audio_data[idx]
        label = self.labels[idx]
        
        # 将文本特征转换为字符串
        text_str = " ".join([str(x) for x in text])
        
        # 对文本进行tokenization
        text_encoding = self.tokenizer(
            text_str,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将音频特征扩展为三维张量 (1, 1, feature_dim)
        audio = np.expand_dims(np.expand_dims(audio, axis=0), axis=0)
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label])
        }

def train(model, train_loader, val_loader, device, num_epochs=10):
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_fused_loss = 0.0
        train_audio_loss = 0.0
        train_text_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            # 获取数据并移动到设备
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # 前向传播
            outputs = model(audio, text_input_ids, text_attention_mask)
            
            # 计算各个损失
            fused_loss = criterion(outputs['fused_logits'], labels)
            audio_loss = criterion(outputs['audio_logits'], labels)
            text_loss = criterion(outputs['text_logits'], labels)
            
            # 总损失
            loss = fused_loss + 0.5 * (audio_loss + text_loss)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item()
            train_fused_loss += fused_loss.item()
            train_audio_loss += audio_loss.item()
            train_text_loss += text_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'fused_loss': fused_loss.item(),
                'audio_loss': audio_loss.item(),
                'text_loss': text_loss.item()
            })
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_fused_loss /= len(train_loader)
        train_audio_loss /= len(train_loader)
        train_text_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_fused_loss = 0.0
        val_audio_loss = 0.0
        val_text_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据并移动到设备
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['label'].squeeze().to(device)
                
                outputs = model(audio, text_input_ids, text_attention_mask)
                
                fused_loss = criterion(outputs['fused_logits'], labels)
                audio_loss = criterion(outputs['audio_logits'], labels)
                text_loss = criterion(outputs['text_logits'], labels)
                
                loss = fused_loss + 0.5 * (audio_loss + text_loss)
                
                val_loss += loss.item()
                val_fused_loss += fused_loss.item()
                val_audio_loss += audio_loss.item()
                val_text_loss += text_loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_fused_loss /= len(val_loader)
        val_audio_loss /= len(val_loader)
        val_text_loss /= len(val_loader)
        
        # 打印训练和验证结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} (Fused: {train_fused_loss:.4f}, Audio: {train_audio_loss:.4f}, Text: {train_text_loss:.4f})')
        print(f'Val Loss: {val_loss:.4f} (Fused: {val_fused_loss:.4f}, Audio: {val_audio_loss:.4f}, Text: {val_text_loss:.4f})')
        
        # 获取并打印融合权重
        weights = model.get_fusion_weights()
        print(f'Fusion Weights - Audio: {weights["audio_weight"]:.3f}, Text: {weights["text_weight"]:.3f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model!')

def main():
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')
    
    # 创建数据集
    data_dir = 'data/CMU_MOSEI/aligned'
    train_dataset = MOSEIDataset(
        os.path.join(data_dir, 'train_text.npy'),
        os.path.join(data_dir, 'train_audio.npy'),
        os.path.join(data_dir, 'train_labels.npy'),
        'train'
    )
    val_dataset = MOSEIDataset(
        os.path.join(data_dir, 'valid_text.npy'),
        os.path.join(data_dir, 'valid_audio.npy'),
        os.path.join(data_dir, 'valid_labels.npy'),
        'valid'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Reduced from 128 to 32
        shuffle=True,
        num_workers=4,  # Reduced from 8 to 4
        prefetch_factor=2,  # Reduced from 4 to 2
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # Reduced from 128 to 32
        shuffle=False,
        num_workers=4,  # Reduced from 8 to 4
        prefetch_factor=2,  # Reduced from 4 to 2
        pin_memory=True
    )
    
    # 创建模型
    model = DecisionFusionModel(
        audio_input_dim=1,  # 因为我们的音频特征是一维的
        hidden_dim=128,
        num_classes=3  # 3个类别：消极、中性、积极
    ).to(device)
    
    # 训练模型
    train(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main() 