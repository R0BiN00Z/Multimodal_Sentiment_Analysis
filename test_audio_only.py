import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models.audio_model import AudioModel  # 需要创建这个模型

class AudioDataset(Dataset):
    def __init__(self, audio_path, label_path, subset_ratio=0.1):
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        
        # 随机选择子集
        total_samples = len(self.labels)
        subset_size = int(total_samples * subset_ratio)
        np.random.seed(42)
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        self.audio_data = self.audio_data[indices]
        self.labels = self.labels[indices]
        
        print(f"\n音频数据集信息 (使用 {subset_ratio*100:.1f}% 的数据):")
        print(f"样本数量: {len(self.labels)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(self.labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx].copy()
        audio = np.expand_dims(np.expand_dims(audio, axis=0), axis=0)
        return {
            'audio': torch.FloatTensor(audio),
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
            audio = batch['audio'].to(device, non_blocking=True)
            labels = batch['label'].squeeze().to(device, non_blocking=True)
            
            outputs = model(audio)
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
    test_dataset = AudioDataset(
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
    model = AudioModel(
        input_dim=1,
        hidden_dim=128,
        num_classes=3
    ).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load('best_audio_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('开始测试...')
    test_loss, accuracy = test(model, test_loader, device)

if __name__ == '__main__':
    main() 