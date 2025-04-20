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

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='train', subset_ratio=0.01):
        self.text_data = np.load(text_path, mmap_mode='r')
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.split = split
        
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
        
        # Class-balanced sampling
        if split == 'train':
            self.balanced_indices = self._get_balanced_indices()
        
        print(f"\n{split} set information (using {subset_ratio*100:.1f}% of data):")
        print(f"Number of samples: {len(self.labels)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def _get_balanced_indices(self):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        max_count = max(counts)
        
        balanced_indices = []
        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
            if len(label_indices) < max_count:
                oversampled_indices = np.random.choice(label_indices, max_count, replace=True)
                balanced_indices.extend(oversampled_indices)
            else:
                balanced_indices.extend(label_indices)
        
        return balanced_indices
    
    def __len__(self):
        if self.split == 'train':
            return len(self.balanced_indices)
        return len(self.text_data)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            idx = self.balanced_indices[idx]
        
        text = self.text_data[idx].copy()
        audio = self.audio_data[idx].copy()
        label = self.labels[idx].copy()
        
        # Convert audio to float32 tensor with shape (1,)
        audio = torch.tensor([float(audio)], dtype=torch.float32)
        
        # Data augmentation: add random noise to audio features during training
        if self.split == 'train':
            noise = torch.randn_like(audio) * 0.01
            audio = audio + noise
        
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
            'audio': audio,
            'label': torch.LongTensor([label])
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

def train(model, train_loader, val_loader, device, num_epochs=20):
    # Mixed precision training
    scaler = amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Early stopping
    early_stopping_patience = 5
    min_delta = 0.001
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Record metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            audio = batch['audio'].to(device, non_blocking=True)
            labels = batch['label'].squeeze().to(device, non_blocking=True)
            
            # Mixed precision training
            with amp.autocast():
                outputs = model(audio, text_input_ids, text_attention_mask)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * train_correct / train_total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad(), amp.autocast():
            for batch in val_loader:
                text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
                text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
                audio = batch['audio'].to(device, non_blocking=True)
                labels = batch['label'].squeeze().to(device, non_blocking=True)
                
                outputs = model(audio, text_input_ids, text_attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, 'best_advanced_fusion_model.pth')
            print(f"Model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve, patience: {patience_counter}/{early_stopping_patience}")
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement in validation loss for {early_stopping_patience} epochs")
            break
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Output final evaluation report
    print("\nFinal evaluation report:")
    print(classification_report(
        all_labels, 
        all_predictions,
        target_names=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    ))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
        return
    
    device = torch.device("cuda")
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    
    # Create datasets
    data_dir = 'data/CMU_MOSEI/aligned'
    subset_ratio = 0.5  # Use 50% of data
    train_dataset = MOSEIDataset(
        os.path.join(data_dir, 'train_text.npy'),
        os.path.join(data_dir, 'train_audio.npy'),
        os.path.join(data_dir, 'train_labels.npy'),
        'train',
        subset_ratio=subset_ratio
    )
    val_dataset = MOSEIDataset(
        os.path.join(data_dir, 'valid_text.npy'),
        os.path.join(data_dir, 'valid_audio.npy'),
        os.path.join(data_dir, 'valid_labels.npy'),
        'valid',
        subset_ratio=subset_ratio
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = AdvancedFusionModel(
        audio_input_dim=1,
        text_hidden_dim=768,
        hidden_dim=512,
        num_classes=5
    ).to(device)
    
    # Train model
    print('Starting training...')
    train(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main() 