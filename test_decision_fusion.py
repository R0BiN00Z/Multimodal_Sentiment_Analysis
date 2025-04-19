import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.decision_fusion_model import DecisionFusionModel

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, subset_ratio=0.1):
        self.text_data = np.load(text_path, mmap_mode='r')
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 随机选择子集
        total_samples = len(self.labels)
        subset_size = int(total_samples * subset_ratio)
        np.random.seed(42)
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        self.text_data = self.text_data[indices]
        self.audio_data = self.audio_data[indices]
        self.labels = self.labels[indices]
        
        print(f"\n测试集信息 (使用 {subset_ratio*100:.1f}% 的数据):")
        print(f"样本数量: {len(self.labels)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(self.labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.text_data[idx].copy()
        audio = self.audio_data[idx].copy()
        label = self.labels[idx].copy()
        
        text_str = " ".join([str(x) for x in text])
        text_encoding = self.tokenizer(
            text_str,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        audio = np.expand_dims(np.expand_dims(audio, axis=0), axis=0)
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label])
        }

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 初始化统计变量
    fused_loss = 0.0
    audio_loss = 0.0
    text_loss = 0.0
    
    fused_correct = 0
    audio_correct = 0
    text_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # 获取数据
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            audio = batch['audio'].to(device, non_blocking=True)
            labels = batch['label'].squeeze().to(device, non_blocking=True)
            
            # 前向传播
            outputs = model(audio, text_input_ids, text_attention_mask)
            
            # 计算各个损失
            fused_loss += criterion(outputs['fused_logits'], labels).item()
            audio_loss += criterion(outputs['audio_logits'], labels).item()
            text_loss += criterion(outputs['text_logits'], labels).item()
            
            # 计算准确率
            _, fused_predicted = torch.max(outputs['fused_logits'].data, 1)
            _, audio_predicted = torch.max(outputs['audio_logits'].data, 1)
            _, text_predicted = torch.max(outputs['text_logits'].data, 1)
            
            total += labels.size(0)
            fused_correct += (fused_predicted == labels).sum().item()
            audio_correct += (audio_predicted == labels).sum().item()
            text_correct += (text_predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    fused_loss /= len(test_loader)
    audio_loss /= len(test_loader)
    text_loss /= len(test_loader)
    
    fused_acc = 100 * fused_correct / total
    audio_acc = 100 * audio_correct / total
    text_acc = 100 * text_correct / total
    
    # 获取融合权重
    weights = model.get_fusion_weights()
    
    # 打印结果
    print('\n测试结果:')
    print('融合模型:')
    print(f'  损失: {fused_loss:.4f}')
    print(f'  准确率: {fused_acc:.2f}%')
    
    print('\n音频模型:')
    print(f'  损失: {audio_loss:.4f}')
    print(f'  准确率: {audio_acc:.2f}%')
    
    print('\n文本模型:')
    print(f'  损失: {text_loss:.4f}')
    print(f'  准确率: {text_acc:.2f}%')
    
    print('\n融合权重:')
    print(f'  音频权重: {weights["audio_weight"]:.3f}')
    print(f'  文本权重: {weights["text_weight"]:.3f}')
    
    return {
        'fused': {'loss': fused_loss, 'accuracy': fused_acc},
        'audio': {'loss': audio_loss, 'accuracy': audio_acc},
        'text': {'loss': text_loss, 'accuracy': text_acc},
        'weights': weights
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查您的GPU设置")
        return
    
    device = torch.device("cuda")
    print(f'使用GPU: {torch.cuda.get_device_name(0)}')
    
    # 创建数据集
    data_dir = 'data/CMU_MOSEI/aligned'
    subset_ratio = 0.1
    test_dataset = MOSEIDataset(
        os.path.join(data_dir, 'test_text.npy'),
        os.path.join(data_dir, 'test_audio.npy'),
        os.path.join(data_dir, 'test_labels.npy'),
        subset_ratio=subset_ratio
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载模型
    model = DecisionFusionModel(
        audio_input_dim=1,
        hidden_dim=128,
        num_classes=3
    ).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('开始测试...')
    results = test(model, test_loader, device)

if __name__ == '__main__':
    main() 