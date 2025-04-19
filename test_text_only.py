import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.text_model import TextModel  # 需要创建这个模型

class TextDataset(Dataset):
    def __init__(self, text_path, label_path, subset_ratio=0.1):
        self.text_data = np.load(text_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 随机选择子集
        total_samples = len(self.labels)
        subset_size = int(total_samples * subset_ratio)
        np.random.seed(42)
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        self.text_data = self.text_data[indices]
        self.labels = self.labels[indices]
        
        print(f"\n文本数据集信息 (使用 {subset_ratio*100:.1f}% 的数据):")
        print(f"样本数量: {len(self.labels)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(self.labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.text_data[idx].copy()
        text_str = " ".join([str(x) for x in text])
        
        text_encoding = self.tokenizer(
            text_str,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'label': torch.LongTensor([self.labels[idx]])
        }

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].squeeze().to(device, non_blocking=True)
            
            outputs = model(text_input_ids, text_attention_mask)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f'\n测试结果:')
    print(f'测试损失: {test_loss:.4f}')
    print(f'准确率: {accuracy:.2f}%')
    
    return test_loss, accuracy

def main():
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查您的GPU设置")
        return
    
    device = torch.device("cuda")
    print(f'使用GPU: {torch.cuda.get_device_name(0)}')
    
    # 创建数据集
    data_dir = 'data/CMU_MOSEI/aligned'
    subset_ratio = 0.1
    test_dataset = TextDataset(
        os.path.join(data_dir, 'test_text.npy'),
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
    model = TextModel(
        hidden_dim=128,
        num_classes=3
    ).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load('best_text_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('开始测试...')
    test_loss, accuracy = test(model, test_loader, device)

if __name__ == '__main__':
    main() 