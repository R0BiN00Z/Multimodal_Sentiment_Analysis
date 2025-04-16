import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multimodal_model import BimodalSentimentModel
from data.dataset import MOSI_Dataset

def train():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_dataset = MOSI_Dataset(split='train')
    val_dataset = MOSI_Dataset(split='valid')
    test_dataset = MOSI_Dataset(split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = BimodalSentimentModel(
        text_input_size=300,  # GloVe 词向量维度
        audio_input_size=74,  # COVAREP 特征维度
        hidden_size=128,
        num_classes=7,  # 7种情感类别
        dropout=0.1
    ).to(device)
    
    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # 获取数据
            text, audio, labels = batch
            text, audio, labels = text.to(device), audio.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(text, audio)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        # 验证
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in val_loader:
                text, audio, labels = batch
                text, audio, labels = text.to(device), audio.to(device), labels.to(device)
                outputs = model(text, audio)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f'Validation Accuracy: {100 * correct / total:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), 'bimodal_model.pth')

if __name__ == '__main__':
    train() 