import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models.audio_model import AudioNet
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MOSEIAudioDataset(Dataset):
    def __init__(self, audio_path, label_path, split='train', subset_ratio=0.01):
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.split = split
        
        # Ensure data alignment
        assert len(self.audio_data) == len(self.labels), \
            f"Data lengths mismatch: audio={len(self.audio_data)}, labels={len(self.labels)}"
        
        # Select subset
        total_samples = len(self.labels)
        subset_size = int(total_samples * subset_ratio)
        np.random.seed(42)
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        # Get subset
        self.audio_data = self.audio_data[indices]
        self.labels = self.labels[indices]
        
        # Print dataset info
        print(f"\n{split} set information:")
        print(f"Number of samples: {len(self.labels)}")
        print(f"Audio data shape: {self.audio_data.shape}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx].copy()
        label = self.labels[idx].copy()
        
        # Convert to PyTorch tensors
        audio = torch.FloatTensor(audio)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'audio': audio,
            'label': label
        }

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*total_correct/total_samples:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * total_correct / total_samples

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch in progress_bar:
            # Move data to device
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(audio)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*total_correct/total_samples:.2f}%'
            })
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, digits=4)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return total_loss / len(val_loader), 100. * total_correct / total_samples, report, conf_matrix

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path='training_metrics.png'):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
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

def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    data_dir = 'data/CMU_MOSEI/aligned'
    train_dataset = MOSEIAudioDataset(
        os.path.join(data_dir, 'train_audio.npy'),
        os.path.join(data_dir, 'train_labels.npy'),
        'train'
    )
    val_dataset = MOSEIAudioDataset(
        os.path.join(data_dir, 'valid_audio.npy'),
        os.path.join(data_dir, 'valid_labels.npy'),
        'valid'
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
    
    # Create model
    model = AudioNet().to(device)
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler)
        
        # Validation phase
        val_loss, val_acc, report, conf_matrix = validate(
            model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f"\nTraining Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print("\nClassification Report:")
        print(report)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_audio_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
            
            # Plot confusion matrix for best model
            plot_confusion_matrix(conf_matrix, 'best_confusion_matrix.png')
    
    # Plot final metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    print("\nTraining completed!")

if __name__ == '__main__':
    main() 