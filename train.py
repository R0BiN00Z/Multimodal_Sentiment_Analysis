import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models import MultimodalModel
from config import ModelConfig
from preprocessing.preprocess_mosei import MOSEIPreprocessor

def train():
    # 初始化配置
    config = ModelConfig()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 初始化预处理器
    preprocessor = MOSEIPreprocessor(config.data_dir)
    
    # 加载数据集
    train_dataset = preprocessor.get_dataset(split='train')
    val_dataset = preprocessor.get_dataset(split='val')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 初始化模型
    model = MultimodalModel(config)
    model.to(config.device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            # 将数据移到设备
            text_inputs = {k: v.to(config.device) for k, v in batch['text'].items()}
            acoustic_features = batch['acoustic'].to(config.device)
            visual_features = batch['visual'].to(config.device)
            labels = batch['label'].to(config.device)
            
            # 前向传播
            logits = model(text_inputs, acoustic_features, visual_features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                text_inputs = {k: v.to(config.device) for k, v in batch['text'].items()}
                acoustic_features = batch['acoustic'].to(config.device)
                visual_features = batch['visual'].to(config.device)
                labels = batch['label'].to(config.device)
                
                logits = model(text_inputs, acoustic_features, visual_features)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 打印训练和验证指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pt'))

if __name__ == '__main__':
    train() 