import matplotlib.pyplot as plt
import numpy as np
import torch
from train_decision_fusion_cuda import train, MOSEIDataset
import os

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, audio_weights, text_weights, save_path='training_curves.png'):
    plt.figure(figsize=(15, 15))
    
    # 创建子图
    plt.subplot(3, 2, 1)
    plt.plot(train_losses['fused'], label='Fused Loss', color='blue')
    plt.plot(train_losses['audio'], label='Audio Loss', color='red')
    plt.plot(train_losses['text'], label='Text Loss', color='green')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(val_losses['fused'], label='Fused Loss', color='blue')
    plt.plot(val_losses['audio'], label='Audio Loss', color='red')
    plt.plot(val_losses['text'], label='Text Loss', color='green')
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(train_accs, label='Training Accuracy', color='blue')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(val_accs, label='Validation Accuracy', color='blue')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(audio_weights, label='Audio Weight', color='red')
    plt.plot(text_weights, label='Text Weight', color='green')
    plt.title('Fusion Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.plot(np.array(audio_weights) + np.array(text_weights), label='Total Weight', color='purple')
    plt.title('Total Fusion Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_training_history(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return {
        'train_losses': {
            'fused': checkpoint.get('train_fused_losses', []),
            'audio': checkpoint.get('train_audio_losses', []),
            'text': checkpoint.get('train_text_losses', [])
        },
        'val_losses': {
            'fused': checkpoint.get('val_fused_losses', []),
            'audio': checkpoint.get('val_audio_losses', []),
            'text': checkpoint.get('val_text_losses', [])
        },
        'train_accs': checkpoint.get('train_accs', []),
        'val_accs': checkpoint.get('val_accs', []),
        'audio_weights': checkpoint.get('audio_weights', []),
        'text_weights': checkpoint.get('text_weights', [])
    }

def main():
    # 检查是否有训练历史记录
    if not os.path.exists('best_model.pth'):
        print("没有找到训练历史记录，请先运行训练脚本")
        return
    
    # 加载训练历史
    history = load_training_history('best_model.pth')
    
    # 绘制训练曲线
    plot_training_curves(
        history['train_losses'],
        history['val_losses'],
        history['train_accs'],
        history['val_accs'],
        history['audio_weights'],
        history['text_weights']
    )
    
    print("训练曲线已保存为 'training_curves.png'")

if __name__ == '__main__':
    main() 