import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models import MultimodalModel
from config import ModelConfig
from preprocessing.preprocess_mosei import MOSEIPreprocessor

def evaluate():
    # 初始化配置
    config = ModelConfig()
    
    # 初始化预处理器
    preprocessor = MOSEIPreprocessor(config.data_dir)
    
    # 加载测试集
    test_dataset = preprocessor.get_dataset(split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 加载模型
    model = MultimodalModel(config)
    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pt')))
    model.to(config.device)
    model.eval()
    
    # 评估
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            text_inputs = {k: v.to(config.device) for k, v in batch['text'].items()}
            acoustic_features = batch['acoustic'].to(config.device)
            visual_features = batch['visual'].to(config.device)
            labels = batch['label'].to(config.device)
            
            logits = model(text_inputs, acoustic_features, visual_features)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    # 计算准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    evaluate() 