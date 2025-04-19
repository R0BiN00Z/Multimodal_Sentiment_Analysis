import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.decision_fusion_model import DecisionFusionModel
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='train', sample_ratio=0.3):
        self.text_data = np.load(text_path)
        self.audio_data = np.load(audio_path)
        self.labels = np.load(label_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 数据采样
        if sample_ratio < 1.0 and split == 'train':
            total_samples = len(self.labels)
            sample_size = int(total_samples * sample_ratio)
            # 确保每个类别都有足够的样本
            indices = []
            for label in range(3):  # 3个类别
                label_indices = np.where(self.labels == label)[0]
                n_samples = int(len(label_indices) * sample_ratio)
                selected = np.random.choice(label_indices, n_samples, replace=False)
                indices.extend(selected)
            
            # 随机打乱
            np.random.shuffle(indices)
            self.text_data = self.text_data[indices]
            self.audio_data = self.audio_data[indices]
            self.labels = self.labels[indices]
        
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
            max_length=256,  # 减小序列长度
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将音频特征扩展为三维张量
        audio = np.expand_dims(np.expand_dims(audio, axis=0), axis=0)
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label])
        }

def train(model, train_loader, val_loader, device, num_epochs=5):
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    
    # 用于记录模态性能
    modality_performance = {
        'train': {'fused': [], 'audio': [], 'text': []},
        'val': {'fused': [], 'audio': [], 'text': []}
    }
    
    # 梯度累积步数
    gradient_accumulation_steps = 4
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_metrics = {
            'loss': 0.0,
            'fused_correct': 0,
            'audio_correct': 0,
            'text_correct': 0,
            'total': 0
        }
        
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
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
            loss = (fused_loss + 0.5 * (audio_loss + text_loss)) / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 计算准确率
            fused_pred = outputs['fused_logits'].argmax(dim=1)
            audio_pred = outputs['audio_logits'].argmax(dim=1)
            text_pred = outputs['text_logits'].argmax(dim=1)
            
            train_metrics['loss'] += loss.item() * gradient_accumulation_steps
            train_metrics['fused_correct'] += (fused_pred == labels).sum().item()
            train_metrics['audio_correct'] += (audio_pred == labels).sum().item()
            train_metrics['text_correct'] += (text_pred == labels).sum().item()
            train_metrics['total'] += labels.size(0)
            
            # 获取当前融合权重
            weights = model.get_fusion_weights()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'fused_acc': train_metrics['fused_correct'] / train_metrics['total'],
                'audio_acc': train_metrics['audio_correct'] / train_metrics['total'],
                'text_acc': train_metrics['text_correct'] / train_metrics['total'],
                'audio_w': f"{weights['audio_weight']:.2f}",
                'text_w': f"{weights['text_weight']:.2f}"
            })
        
        # 计算训练指标
        train_loss = train_metrics['loss'] / len(train_loader)
        train_fused_acc = train_metrics['fused_correct'] / train_metrics['total']
        train_audio_acc = train_metrics['audio_correct'] / train_metrics['total']
        train_text_acc = train_metrics['text_correct'] / train_metrics['total']
        
        # 验证阶段
        model.eval()
        val_metrics = {
            'loss': 0.0,
            'fused_correct': 0,
            'audio_correct': 0,
            'text_correct': 0,
            'total': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['label'].squeeze().to(device)
                
                outputs = model(audio, text_input_ids, text_attention_mask)
                
                fused_loss = criterion(outputs['fused_logits'], labels)
                audio_loss = criterion(outputs['audio_logits'], labels)
                text_loss = criterion(outputs['text_logits'], labels)
                
                loss = fused_loss + 0.5 * (audio_loss + text_loss)
                
                # 计算准确率
                fused_pred = outputs['fused_logits'].argmax(dim=1)
                audio_pred = outputs['audio_logits'].argmax(dim=1)
                text_pred = outputs['text_logits'].argmax(dim=1)
                
                val_metrics['loss'] += loss.item()
                val_metrics['fused_correct'] += (fused_pred == labels).sum().item()
                val_metrics['audio_correct'] += (audio_pred == labels).sum().item()
                val_metrics['text_correct'] += (text_pred == labels).sum().item()
                val_metrics['total'] += labels.size(0)
        
        # 计算验证指标
        val_loss = val_metrics['loss'] / len(val_loader)
        val_fused_acc = val_metrics['fused_correct'] / val_metrics['total']
        val_audio_acc = val_metrics['audio_correct'] / val_metrics['total']
        val_text_acc = val_metrics['text_correct'] / val_metrics['total']
        
        # 记录性能
        modality_performance['train']['fused'].append(train_fused_acc)
        modality_performance['train']['audio'].append(train_audio_acc)
        modality_performance['train']['text'].append(train_text_acc)
        modality_performance['val']['fused'].append(val_fused_acc)
        modality_performance['val']['audio'].append(val_audio_acc)
        modality_performance['val']['text'].append(val_text_acc)
        
        # 打印训练和验证结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Fused Acc: {train_fused_acc:.4f}, Audio Acc: {train_audio_acc:.4f}, Text Acc: {train_text_acc:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, Fused Acc: {val_fused_acc:.4f}, Audio Acc: {val_audio_acc:.4f}, Text Acc: {val_text_acc:.4f}')
        
        # 获取并打印融合权重
        weights = model.get_fusion_weights()
        print(f'Fusion Weights - Audio: {weights["audio_weight"]:.3f}, Text: {weights["text_weight"]:.3f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'modality_performance': modality_performance
            }, 'best_model.pth')
            print('Saved best model!')

    return modality_performance

def evaluate_model(model, test_loader, device):
    """在测试集上评估模型性能"""
    model.eval()
    metrics = {
        'loss': 0.0,
        'fused_correct': 0,
        'audio_correct': 0,
        'text_correct': 0,
        'total': 0,
        'confusion_matrix': {
            'fused': np.zeros((3, 3)),
            'audio': np.zeros((3, 3)),
            'text': np.zeros((3, 3))
        }
    }
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            outputs = model(audio, text_input_ids, text_attention_mask)
            
            fused_loss = criterion(outputs['fused_logits'], labels)
            audio_loss = criterion(outputs['audio_logits'], labels)
            text_loss = criterion(outputs['text_logits'], labels)
            
            loss = fused_loss + 0.5 * (audio_loss + text_loss)
            
            # 计算预测结果
            fused_pred = outputs['fused_logits'].argmax(dim=1)
            audio_pred = outputs['audio_logits'].argmax(dim=1)
            text_pred = outputs['text_logits'].argmax(dim=1)
            
            # 更新指标
            metrics['loss'] += loss.item()
            metrics['fused_correct'] += (fused_pred == labels).sum().item()
            metrics['audio_correct'] += (audio_pred == labels).sum().item()
            metrics['text_correct'] += (text_pred == labels).sum().item()
            metrics['total'] += labels.size(0)
            
            # 更新混淆矩阵
            for i, j in zip(labels.cpu().numpy(), fused_pred.cpu().numpy()):
                metrics['confusion_matrix']['fused'][i][j] += 1
            for i, j in zip(labels.cpu().numpy(), audio_pred.cpu().numpy()):
                metrics['confusion_matrix']['audio'][i][j] += 1
            for i, j in zip(labels.cpu().numpy(), text_pred.cpu().numpy()):
                metrics['confusion_matrix']['text'][i][j] += 1
    
    # 计算各项指标
    results = {
        'loss': metrics['loss'] / len(test_loader),
        'fused_accuracy': metrics['fused_correct'] / metrics['total'],
        'audio_accuracy': metrics['audio_correct'] / metrics['total'],
        'text_accuracy': metrics['text_correct'] / metrics['total'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    # 计算每个类别的 F1 分数
    for modality in ['fused', 'audio', 'text']:
        cm = metrics['confusion_matrix'][modality]
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        results[f'{modality}_f1'] = {
            'negative': f1[0],
            'neutral': f1[1],
            'positive': f1[2]
        }
    
    return results

def plot_training_process(modality_performance, save_dir='plots'):
    """Plot training process visualization"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(modality_performance['train']['fused']) + 1)
    
    # Training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, modality_performance['train']['fused'], 'b-', label='Fusion Model')
    plt.plot(epochs, modality_performance['train']['audio'], 'r--', label='Audio Only')
    plt.plot(epochs, modality_performance['train']['text'], 'g-.', label='Text Only')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, modality_performance['val']['fused'], 'b-', label='Fusion Model')
    plt.plot(epochs, modality_performance['val']['audio'], 'r--', label='Audio Only')
    plt.plot(epochs, modality_performance['val']['text'], 'g-.', label='Text Only')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_process.png'))
    plt.close()

def plot_test_results(test_results, save_dir='plots'):
    """Plot test results visualization"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(10, 6))
    accuracies = [
        test_results['fused_accuracy'],
        test_results['audio_accuracy'],
        test_results['text_accuracy']
    ]
    plt.bar(['Fusion Model', 'Audio Only', 'Text Only'], accuracies)
    plt.title('Accuracy Comparison Across Modalities')
    plt.ylabel('Accuracy')
    plt.grid(True, axis='y')
    for i, v in enumerate(accuracies):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. F1 score comparison
    plt.figure(figsize=(12, 6))
    modalities = ['fused', 'audio', 'text']
    labels = ['Negative', 'Neutral', 'Positive']
    x = np.arange(len(labels))
    width = 0.25
    
    for i, modality in enumerate(modalities):
        f1_scores = [
            test_results[f'{modality}_f1']['negative'],
            test_results[f'{modality}_f1']['neutral'],
            test_results[f'{modality}_f1']['positive']
        ]
        plt.bar(x + i*width, f1_scores, width, 
                label=['Fusion Model', 'Audio Only', 'Text Only'][i])
    
    plt.xlabel('Sentiment Category')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores by Modality and Sentiment')
    plt.xticks(x + width, labels)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'f1_scores.png'))
    plt.close()
    
    # 3. Confusion matrix heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Fusion Model', 'Audio Only', 'Text Only']
    
    for i, (modality, title) in enumerate(zip(['fused', 'audio', 'text'], titles)):
        cm = test_results['confusion_matrix'][modality]
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{title} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        axes[i].set_xticklabels(['Negative', 'Neutral', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Neutral', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.close()

def main():
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')
    
    # 创建数据集
    data_dir = 'data/CMU_MOSEI/aligned'
    train_dataset = MOSEIDataset(
        os.path.join(data_dir, 'train_text.npy'),
        os.path.join(data_dir, 'train_audio.npy'),
        os.path.join(data_dir, 'train_labels.npy'),
        'train',
        sample_ratio=0.3  # 使用30%训练数据
    )
    val_dataset = MOSEIDataset(
        os.path.join(data_dir, 'valid_text.npy'),
        os.path.join(data_dir, 'valid_audio.npy'),
        os.path.join(data_dir, 'valid_labels.npy'),
        'valid',
        sample_ratio=0.3  # 使用30%验证数据
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 减小批处理大小
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # 减小批处理大小
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = DecisionFusionModel(
        audio_input_dim=1,  # 因为我们的音频特征是一维的
        hidden_dim=128,
        num_classes=3  # 3个类别：消极、中性、积极
    ).to(device)
    
    # 打印模型信息
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    modality_performance = train(model, train_loader, val_loader, device)
    
    # 绘制训练过程图表
    plot_training_process(modality_performance)
    
    # 创建测试数据集
    test_dataset = MOSEIDataset(
        os.path.join(data_dir, 'test_text.npy'),
        os.path.join(data_dir, 'test_audio.npy'),
        os.path.join(data_dir, 'test_labels.npy'),
        'test',
        sample_ratio=1.0  # 测试集使用全部数据
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载最佳模型
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['val_loss']
    modality_performance = checkpoint['modality_performance']
    
    # 评估模型
    test_results = evaluate_model(model, test_loader, device)
    
    # 绘制测试结果图表
    plot_test_results(test_results)
    
    print("\nPlots have been saved in 'plots' directory:")
    print("1. training_process.png - Training Process Visualization")
    print("2. accuracy_comparison.png - Accuracy Comparison")
    print("3. f1_scores.png - F1 Score Comparison")
    print("4. confusion_matrices.png - Confusion Matrices")

if __name__ == '__main__':
    main() 