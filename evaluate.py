import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from models.multimodal_model import MultimodalModel

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='test'):
        # 使用 mmap_mode='r' 来减少内存使用
        self.text_data = np.load(text_path, mmap_mode='r')
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 确保数据对齐
        assert len(self.text_data) == len(self.audio_data) == len(self.labels), \
            f"Data lengths mismatch: text={len(self.text_data)}, audio={len(self.audio_data)}, labels={len(self.labels)}"
        
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
        text = self.text_data[idx].copy()
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

def evaluate():
    print("开始评估...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = MultimodalModel(audio_input_dim=1, hidden_dim=128, num_classes=5)  # 修改音频输入维度
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型训练了 {checkpoint['epoch']} 个epoch")
    print(f"最佳验证损失: {checkpoint['loss']:.4f}")
    
    # 加载测试数据
    print("加载测试数据...")
    test_dataset = MOSEIDataset(
        text_path='data/CMU_MOSEI/aligned/test_text.npy',
        audio_path='data/CMU_MOSEI/aligned/test_audio.npy',
        label_path='data/CMU_MOSEI/aligned/test_labels.npy'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 评估
    all_predictions = []
    all_labels = []
    
    print("开始评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估进度"):
            # 获取数据并移动到设备
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # 前向传播
            outputs = model(audio, text_input_ids, text_attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            
            # 收集预测结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    print("\n分类报告:")
    print(classification_report(
        all_labels, 
        all_predictions,
        target_names=['非常消极', '消极', '中性', '积极', '非常积极']
    ))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(all_labels, all_predictions))
    
    # 计算总体准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\n总体准确率: {accuracy:.4f}")

if __name__ == '__main__':
    evaluate() 