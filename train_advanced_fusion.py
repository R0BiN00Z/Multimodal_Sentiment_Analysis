import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.advanced_fusion_model import AdvancedFusionModel
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# Training hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 16  # Reduced batch size
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 2000
GRADIENT_CLIP = 1.0

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='train', subset_ratio=0.01):
        self.split = split
        
        # Load data
        self.text_data = np.load(text_path)
        self.audio_data = np.load(audio_path)
        self.labels = np.load(label_path)
        
        # Ensure data alignment
        assert len(self.text_data) == len(self.audio_data) == len(self.labels), \
            f"Data lengths mismatch: text={len(self.text_data)}, audio={len(self.audio_data)}, labels={len(self.labels)}"
        
        # Select subset
        total_samples = len(self.labels)
        subset_size = int(total_samples * subset_ratio)
        np.random.seed(42)
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        # Get subset
        self.text_data = self.text_data[indices]
        self.audio_data = self.audio_data[indices]
        self.labels = self.labels[indices]
        
        # Clean and normalize text data
        self.text_data = np.nan_to_num(self.text_data, nan=0.0, posinf=1.0, neginf=-1.0)
        text_mean = np.mean(self.text_data, axis=(0, 1), keepdims=True)
        text_std = np.std(self.text_data, axis=(0, 1), keepdims=True) + 1e-6
        self.text_data = (self.text_data - text_mean) / text_std
        
        # Clean and normalize audio data
        self.audio_data = np.nan_to_num(self.audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_mean = np.mean(self.audio_data, axis=(0, 1), keepdims=True)
        audio_std = np.std(self.audio_data, axis=(0, 1), keepdims=True) + 1e-6
        self.audio_data = (self.audio_data - audio_mean) / audio_std
        
        # Clip values to reasonable range
        self.text_data = np.clip(self.text_data, -10.0, 10.0)
        self.audio_data = np.clip(self.audio_data, -10.0, 10.0)
        
        # Adjust audio feature dimension order
        self.audio_data = np.transpose(self.audio_data, (0, 2, 1))
        
        # Print data information
        print(f"\n{split} set information (using {subset_ratio*100:.1f}% of data):")
        print(f"Number of samples: {len(self.labels)}")
        print(f"Text data shape: {self.text_data.shape}")
        print(f"Audio data shape: {self.audio_data.shape}")
        print(f"Text data range: [{self.text_data.min():.3f}, {self.text_data.max():.3f}]")
        print(f"Audio data range: [{self.audio_data.min():.3f}, {self.audio_data.max():.3f}]")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Label distribution:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples ({c/len(self.labels)*100:.2f}%)")
        
        if split == 'train':
            self.balanced_indices = self._get_balanced_indices()
    
    def _get_balanced_indices(self):
        # 获取每个类别的样本数
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        max_count = max(counts)
        
        balanced_indices = []
        for label in unique_labels:
            # 获取当前类别的所有样本索引
            label_indices = np.where(self.labels == label)[0]
            # 如果样本数不足，进行过采样
            if len(label_indices) < max_count:
                oversampled_indices = np.random.choice(label_indices, max_count, replace=True)
                balanced_indices.extend(oversampled_indices)
            else:
                # 如果样本数过多，进行欠采样
                undersampled_indices = np.random.choice(label_indices, max_count, replace=False)
                balanced_indices.extend(undersampled_indices)
        
        return balanced_indices
    
    def __len__(self):
        if self.split == 'train':
            return len(self.balanced_indices)
        return len(self.text_data)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            idx = self.balanced_indices[idx]
        
        text = self.text_data[idx]
        audio = self.audio_data[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        text = torch.FloatTensor(text)
        audio = torch.FloatTensor(audio)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'text_input_ids': text,
            'text_attention_mask': torch.ones_like(text[:, 0]),
            'audio': audio,
            'label': label
        }

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_path='training_metrics.png'):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_class_weights(labels):
    # 计算类别权重
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
    return class_weights

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    total_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        text_data = batch['text_input_ids'].to(device)
        audio_data = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        # Skip empty batches
        if labels.numel() == 0:
            continue
            
        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            logits = model(audio_data, text_data)
            loss = criterion(logits, labels)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            print(f"NaN detected in logits at batch {batch_idx}")
            continue
        
        # Backward pass with AMP
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')
    
    # Calculate epoch metrics
    if total_batches > 0:
        avg_loss = total_loss / total_batches
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        avg_loss = float('inf')
        accuracy = 0.0
        f1 = 0.0
    
    return avg_loss, accuracy, f1

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text_data = batch['text_input_ids'].to(device)
            audio_data = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            if labels.numel() == 0:
                continue
                
            with torch.amp.autocast('cuda'):
                logits = model(audio_data, text_data)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_batches += 1
    
    if total_batches > 0:
        avg_loss = total_loss / total_batches
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        avg_loss = float('inf')
        accuracy = 0.0
        f1 = 0.0
    
    return avg_loss, accuracy, f1

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize AMP scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Load datasets
    train_dataset = MOSEIDataset(
        text_path='data/CMU_MOSEI/aligned/train_text.npy',
        audio_path='data/CMU_MOSEI/aligned/train_audio.npy',
        label_path='data/CMU_MOSEI/aligned/train_labels.npy',
        split='train',
        subset_ratio=0.5
    )
    
    val_dataset = MOSEIDataset(
        text_path='data/CMU_MOSEI/aligned/valid_text.npy',
        audio_path='data/CMU_MOSEI/aligned/valid_audio.npy',
        label_path='data/CMU_MOSEI/aligned/valid_labels.npy',
        split='valid',
        subset_ratio=0.5
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = AdvancedFusionModel(
        audio_input_dim=74,
        text_input_dim=300,
        hidden_dim=256,
        num_classes=5
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.5,
        anneal_strategy='cos',
        div_factor=1.5,
        final_div_factor=10.0
    )
    
    # Training loop
    best_val_f1 = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, 'best_model.pt')
            print(f'New best model saved with validation F1: {best_val_f1:.4f}')
        
        scheduler.step()
    
    print('Training completed!')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    main() 