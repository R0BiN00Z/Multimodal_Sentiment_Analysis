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
import torch.cuda.amp as amp

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='train', subset_ratio=0.1):
        # 使用 mmap_mode='r' 来减少内存使用
        self.text_data = np.load(text_path, mmap_mode='r')
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 确保数据对齐
        assert len(self.text_data) == len(self.audio_data) == len(self.labels), \
            f"Data lengths mismatch: text={len(self.text_data)}, audio={len(self.audio_data)}, labels={len(self.labels)}"
        
        # 随机选择子集
        total_samples = len(self.labels)
        subset_size = int(total_samples * subset_ratio)
        np.random.seed(42)  # 设置随机种子以确保可重复性
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        # 获取子集
        self.text_data = self.text_data[indices]
        self.audio_data = self.audio_data[indices]
        self.labels = self.labels[indices]
        
        # 打印数据集信息
        print(f"\n{split} 集信息 (使用 {subset_ratio*100:.1f}% 的数据):")
        print(f"样本数量: {len(self.labels)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(self.labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        # 获取文本和音频特征
        text = self.text_data[idx].copy()  # 使用copy避免mmap问题
        audio = self.audio_data[idx].copy()
        label = self.labels[idx].copy()
        
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
    # 使用混合精度训练
    scaler = amp.GradScaler()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # 初始化早停参数
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # 初始化记录列表
    train_fused_losses = []
    train_audio_losses = []
    train_text_losses = []
    val_fused_losses = []
    val_audio_losses = []
    val_text_losses = []
    train_accs = []
    val_accs = []
    audio_weights = []
    text_weights = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_fused_loss = 0.0
        train_audio_loss = 0.0
        train_text_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据并移动到设备
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            audio = batch['audio'].to(device, non_blocking=True)
            labels = batch['label'].squeeze().to(device, non_blocking=True)
            
            # 使用混合精度训练
            with amp.autocast():
                # 前向传播
                outputs = model(audio, text_input_ids, text_attention_mask)
                
                # 计算各个损失
                fused_loss = criterion(outputs['fused_logits'], labels)
                audio_loss = criterion(outputs['audio_logits'], labels)
                text_loss = criterion(outputs['text_logits'], labels)
                
                # 总损失
                loss = fused_loss + 0.5 * (audio_loss + text_loss)
            
            # 反向传播和优化
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            # 更新统计信息
            train_fused_loss += fused_loss.item()
            train_audio_loss += audio_loss.item()
            train_text_loss += text_loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs['fused_logits'].data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'fused_loss': fused_loss.item(),
                'audio_loss': audio_loss.item(),
                'text_loss': text_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # 清除缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # 计算平均训练损失
        train_fused_loss /= len(train_loader)
        train_audio_loss /= len(train_loader)
        train_text_loss /= len(train_loader)
        
        # 记录训练损失
        train_fused_losses.append(train_fused_loss)
        train_audio_losses.append(train_audio_loss)
        train_text_losses.append(train_text_loss)
        
        # 计算训练准确率
        train_acc = 100 * train_correct / train_total
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_fused_loss = 0.0
        val_audio_loss = 0.0
        val_text_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(), amp.autocast():
            for batch in val_loader:
                # 获取数据并移动到设备
                text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
                text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
                audio = batch['audio'].to(device, non_blocking=True)
                labels = batch['label'].squeeze().to(device, non_blocking=True)
                
                outputs = model(audio, text_input_ids, text_attention_mask)
                
                fused_loss = criterion(outputs['fused_logits'], labels)
                audio_loss = criterion(outputs['audio_logits'], labels)
                text_loss = criterion(outputs['text_logits'], labels)
                
                loss = fused_loss + 0.5 * (audio_loss + text_loss)
                
                val_fused_loss += fused_loss.item()
                val_audio_loss += audio_loss.item()
                val_text_loss += text_loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs['fused_logits'].data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算平均验证损失
        val_fused_loss /= len(val_loader)
        val_audio_loss /= len(val_loader)
        val_text_loss /= len(val_loader)
        
        # 记录验证损失
        val_fused_losses.append(val_fused_loss)
        val_audio_losses.append(val_audio_loss)
        val_text_losses.append(val_text_loss)
        
        # 计算验证准确率
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        
        # 获取并记录融合权重
        weights = model.get_fusion_weights()
        audio_weights.append(weights['audio_weight'])
        text_weights.append(weights['text_weight'])
        
        # 打印训练和验证结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Losses:')
        print(f'  Fused: {train_fused_loss:.4f}')
        print(f'  Audio: {train_audio_loss:.4f}')
        print(f'  Text: {train_text_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Val Losses:')
        print(f'  Fused: {val_fused_loss:.4f}')
        print(f'  Audio: {val_audio_loss:.4f}')
        print(f'  Text: {val_text_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.2f}%')
        print(f'Fusion Weights - Audio: {weights["audio_weight"]:.3f}, Text: {weights["text_weight"]:.3f}')
        
        # 保存最佳模型
        if val_fused_loss < best_val_loss:
            best_val_loss = val_fused_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_fused_loss,
                'train_fused_losses': train_fused_losses,
                'train_audio_losses': train_audio_losses,
                'train_text_losses': train_text_losses,
                'val_fused_losses': val_fused_losses,
                'val_audio_losses': val_audio_losses,
                'val_text_losses': val_text_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'audio_weights': audio_weights,
                'text_weights': text_weights
            }, 'best_model.pth')
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    return {
        'train_losses': {
            'fused': train_fused_losses,
            'audio': train_audio_losses,
            'text': train_text_losses
        },
        'val_losses': {
            'fused': val_fused_losses,
            'audio': val_audio_losses,
            'text': val_text_losses
        },
        'train_accs': train_accs,
        'val_accs': val_accs,
        'audio_weights': audio_weights,
        'text_weights': text_weights
    }

def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查您的GPU设置")
        return
    
    # 设置设备
    device = torch.device("cuda")
    print(f'使用GPU: {torch.cuda.get_device_name(0)}')
    
    # 创建数据集
    data_dir = 'data/CMU_MOSEI/aligned'
    subset_ratio = 0.01  # 修改为1%的数据
    train_dataset = MOSEIDataset(
        os.path.join(data_dir, 'train_text.npy'),
        os.path.join(data_dir, 'train_audio.npy'),
        os.path.join(data_dir, 'train_labels.npy'),
        'train',
        subset_ratio=subset_ratio
    )
    val_dataset = MOSEIDataset(
        os.path.join(data_dir, 'valid_text.npy'),
        os.path.join(data_dir, 'valid_audio.npy'),
        os.path.join(data_dir, 'valid_labels.npy'),
        'valid',
        subset_ratio=subset_ratio
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # 创建模型
    model = DecisionFusionModel(
        audio_input_dim=1,
        hidden_dim=128,
        num_classes=5
    ).to(device)
    
    # 训练模型
    print('Starting training...')
    train(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main() 