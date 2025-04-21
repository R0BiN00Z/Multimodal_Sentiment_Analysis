import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.decision_fusion_model import DecisionFusionModel
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F

class MOSEIDataset(Dataset):
    def __init__(self, text_path, audio_path, label_path, split='train', subset_ratio=0.01):
        self.text_data = np.load(text_path, mmap_mode='r')
        self.audio_data = np.load(audio_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
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
        print(f"Text data shape: {self.text_data.shape}")
        print(f"Audio data shape: {self.audio_data.shape}")
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
        
        # Convert to PyTorch tensors
        text = torch.FloatTensor(text)
        audio = torch.FloatTensor(audio)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'text': text,
            'audio': audio,
            'label': label
        }

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, 
                         audio_losses, text_losses, audio_accs, text_accs,
                         fusion_weights, save_path='training_metrics.png'):
    plt.figure(figsize=(20, 15))
    
    # Plot total loss
    plt.subplot(3, 2, 1)
    plt.plot(train_losses, label='Train Loss (Total)')
    plt.plot(val_losses, label='Val Loss (Total)')
    plt.title('Total Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot audio loss
    plt.subplot(3, 2, 2)
    plt.plot(audio_losses['train'], label='Train Loss (Audio)')
    plt.plot(audio_losses['val'], label='Val Loss (Audio)')
    plt.title('Audio Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot text loss
    plt.subplot(3, 2, 3)
    plt.plot(text_losses['train'], label='Train Loss (Text)')
    plt.plot(text_losses['val'], label='Val Loss (Text)')
    plt.title('Text Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot total accuracy
    plt.subplot(3, 2, 4)
    plt.plot(train_accs, label='Train Accuracy (Total)')
    plt.plot(val_accs, label='Val Accuracy (Total)')
    plt.title('Total Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot audio accuracy
    plt.subplot(3, 2, 5)
    plt.plot(audio_accs['train'], label='Train Accuracy (Audio)')
    plt.plot(audio_accs['val'], label='Val Accuracy (Audio)')
    plt.title('Audio Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot text accuracy
    plt.subplot(3, 2, 6)
    plt.plot(text_accs['train'], label='Train Accuracy (Text)')
    plt.plot(text_accs['val'], label='Val Accuracy (Text)')
    plt.title('Text Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Hyperparameters
LEARNING_RATE = 1e-5  # Reduced learning rate
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 30
PATIENCE = 5  # For early stopping
GRAD_CLIP = 5.0  # Increased gradient clipping threshold

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=5, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        
        # Create mask for valid (not ignored) positions
        valid_mask = (target != self.ignore_index)
        valid_positions = valid_mask.sum()
        
        if valid_positions == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Apply log_softmax with better numerical stability
        log_prob = F.log_softmax(pred, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Calculate loss only for valid positions
        losses = -(true_dist * log_prob)
        return losses[valid_mask].mean()

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    audio_correct = 0
    text_correct = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Log input shapes and ranges periodically
        if batch_idx % 100 == 0:
            print(f'\nBatch {batch_idx} input stats:')
            print(f'Audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]')
            print(f'Text shape: {text.shape}, range: [{text.min():.3f}, {text.max():.3f}]')
            print(f'Labels shape: {labels.shape}, unique values: {torch.unique(labels).tolist()}')
        
        optimizer.zero_grad()
        
        # Forward pass with gradient scaling
        with torch.amp.autocast('cuda'):
            outputs, text_logits, audio_logits = model(audio, text)
            
            # Calculate losses with stability checks
            try:
                loss = criterion(outputs, labels)
                audio_loss = criterion(audio_logits, labels)
                text_loss = criterion(text_logits, labels)
                
                # Combined loss with stability check
                if not (torch.isnan(loss) or torch.isnan(audio_loss) or torch.isnan(text_loss)):
                    total_loss = loss + 0.5 * (audio_loss + text_loss)
                else:
                    print(f"Warning: NaN loss detected at batch {batch_idx}")
                    continue
                
            except RuntimeError as e:
                print(f"Error in loss calculation at batch {batch_idx}: {e}")
                continue
            
            # Log intermediate outputs
            if batch_idx % 100 == 0:
                print('\nModel outputs:')
                print(f'Fused logits shape: {outputs.shape}, range: [{outputs.min():.3f}, {outputs.max():.3f}]')
                print(f'Text logits shape: {text_logits.shape}, range: [{text_logits.min():.3f}, {text_logits.max():.3f}]')
                print(f'Audio logits shape: {audio_logits.shape}, range: [{audio_logits.min():.3f}, {audio_logits.max():.3f}]')
                print(f'Loss components - Main: {loss:.4f}, Audio: {audio_loss:.4f}, Text: {text_loss:.4f}')
                
                weights = model.get_fusion_weights()
                print(f'Fusion weights - Audio: {weights["audio_weight"]:.3f}, Text: {weights["text_weight"]:.3f}')
        
        # Backward pass with gradient clipping
        total_loss.backward()
        
        # Log gradients
        if batch_idx % 100 == 0:
            print('\nGradient stats before clipping:')
            total_grad_norm = 0
            max_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    if grad_norm > 10:
                        print(f'Large gradient in {name}: {grad_norm:.3f}')
            print(f'Total gradient norm: {total_grad_norm:.3f}')
            print(f'Max gradient norm: {max_grad_norm:.3f}')
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Log gradients after clipping
        if batch_idx % 100 == 0:
            print('\nGradient stats after clipping:')
            total_grad_norm = 0
            max_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    max_grad_norm = max(max_grad_norm, grad_norm)
            print(f'Total gradient norm: {total_grad_norm:.3f}')
            print(f'Max gradient norm: {max_grad_norm:.3f}')
        
        # Update weights
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)
            
            _, audio_pred = audio_logits.max(1)
            _, text_pred = text_logits.max(1)
            audio_correct += audio_pred.eq(labels).sum().item()
            text_correct += text_pred.eq(labels).sum().item()
        
        # Log batch metrics
        if batch_idx % 100 == 0:
            print(f'\nBatch {batch_idx}/{len(train_loader)} metrics:')
            print(f'Loss: {total_loss.item():.4f}')
            print(f'Batch accuracy: {100.*batch_correct/labels.size(0):.2f}%')
            print(f'Running accuracy: {100.*total_correct/total_samples:.2f}%')
            print(f'Audio accuracy: {100.*audio_correct/total_samples:.2f}%')
            print(f'Text accuracy: {100.*text_correct/total_samples:.2f}%')
            
            # Log prediction distribution
            pred_dist = torch.bincount(predicted, minlength=5)
            print(f'Prediction distribution: {pred_dist.tolist()}')
    
    return total_loss.item() / len(train_loader), 100.*total_correct/total_samples, \
           100.*audio_correct/total_samples, 100.*text_correct/total_samples

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    audio_correct = 0
    text_correct = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Log input shapes and ranges periodically
            if batch_idx % 50 == 0:
                print(f'\nValidation batch {batch_idx} input stats:')
                print(f'Audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]')
                print(f'Text shape: {text.shape}, range: [{text.min():.3f}, {text.max():.3f}]')
                print(f'Labels shape: {labels.shape}, unique values: {torch.unique(labels).tolist()}')
            
            with torch.amp.autocast('cuda'):
                outputs, text_logits, audio_logits = model(audio, text)
                loss = criterion(outputs, labels)
                
                # Log outputs periodically
                if batch_idx % 50 == 0:
                    print('\nValidation outputs:')
                    print(f'Fused logits range: [{outputs.min():.3f}, {outputs.max():.3f}]')
                    print(f'Text logits range: [{text_logits.min():.3f}, {text_logits.max():.3f}]')
                    print(f'Audio logits range: [{audio_logits.min():.3f}, {audio_logits.max():.3f}]')
                    weights = model.get_fusion_weights()
                    print(f'Fusion weights - Audio: {weights["audio_weight"]:.3f}, Text: {weights["text_weight"]:.3f}')
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            # Calculate individual modality accuracies
            _, audio_pred = audio_logits.max(1)
            _, text_pred = text_logits.max(1)
            audio_correct += audio_pred.eq(labels).sum().item()
            text_correct += text_pred.eq(labels).sum().item()
            
            # Log metrics periodically
            if batch_idx % 50 == 0:
                print(f'\nValidation batch {batch_idx} metrics:')
                print(f'Loss: {loss.item():.4f}')
                print(f'Batch accuracy: {100.*predicted.eq(labels).sum().item()/labels.size(0):.2f}%')
                print(f'Running accuracy: {100.*total_correct/total_samples:.2f}%')
                print(f'Audio accuracy: {100.*audio_correct/total_samples:.2f}%')
                print(f'Text accuracy: {100.*text_correct/total_samples:.2f}%')
                
                # Log prediction distribution
                pred_dist = torch.bincount(predicted, minlength=5)
                print(f'Prediction distribution: {pred_dist.tolist()}')
    
    return total_loss / len(val_loader), 100.*total_correct/total_samples, \
           100.*audio_correct/total_samples, 100.*text_correct/total_samples

def train(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):
    # Initialize criterion with label smoothing
    criterion = LabelSmoothingLoss(classes=5, smoothing=0.1)
    
    # Initialize optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warm-up
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warm-up for 10% of training
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10.0,
        final_div_factor=1e4,
    )
    
    train_losses = []
    val_losses = []
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.2e}')
        
        # Training phase
        train_loss, train_acc, train_audio_acc, train_text_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validation phase
        val_loss, val_acc, val_audio_acc, val_text_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'\nEpoch {epoch+1} Results:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Train Audio Acc: {train_audio_acc:.2f}%, Train Text Acc: {train_text_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Audio Acc: {val_audio_acc:.2f}%, Val Text Acc: {val_text_acc:.2f}%')
        
        # Early stopping with model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                # Load best model
                checkpoint = torch.load('best_model.pth')
                model.load_state_dict(checkpoint['model_state_dict'])
                break
    
    return train_losses, val_losses

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
        return
    
    device = torch.device("cuda")
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create datasets
    data_dir = 'data/CMU_MOSEI/aligned'
    subset_ratio = 0.01  # Use 1% of data
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
    
    # Create data loaders without multiprocessing on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No multiprocessing on Windows
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # No multiprocessing on Windows
        pin_memory=True
    )
    
    # Create model
    model = DecisionFusionModel(
        audio_input_dim=74,
        text_input_dim=300,
        hidden_dim=256,
        num_classes=5
    ).to(device)
    
    # Print model architecture and parameter count
    print('\nModel Architecture:')
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Train model
    print('\nStarting training...')
    train(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main() 