import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.text_model import TextTransformer
from sklearn.metrics import accuracy_score, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

def normalize_features(features):
    """Normalize text features"""
    # Replace inf values with nan
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Print initial stats
    logger.info(f"Initial data stats:")
    logger.info(f"Mean: {np.mean(features):.4f}")
    logger.info(f"Std: {np.std(features):.4f}")
    logger.info(f"Min: {np.min(features):.4f}")
    logger.info(f"Max: {np.max(features):.4f}")
    
    # Normalize each feature independently
    for i in range(features.shape[1]):
        for j in range(features.shape[2]):
            feat = features[:, i, j]
            
            # Skip if all values are 0
            if np.all(feat == 0):
                continue
            
            # Standardize with epsilon to prevent division by zero
            mean = np.mean(feat)
            std = np.std(feat) + 1e-6
            feat = (feat - mean) / std
            
            features[:, i, j] = feat
    
    # Print final stats
    logger.info(f"\nNormalized data stats:")
    logger.info(f"Mean: {np.mean(features):.4f}")
    logger.info(f"Std: {np.std(features):.4f}")
    logger.info(f"Min: {np.min(features):.4f}")
    logger.info(f"Max: {np.max(features):.4f}")
    
    return features.astype(np.float32)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (text, labels) in enumerate(train_loader):
        text = text.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(text)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, labels in val_loader:
            text = text.to(device)
            labels = labels.to(device)
            
            outputs = model(text)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load and preprocess data
    logger.info("\nLoading and preprocessing data...")
    train_text = np.load('data/CMU_MOSEI/aligned/train_text.npy')
    train_labels = np.load('data/CMU_MOSEI/aligned/train_labels.npy')
    val_text = np.load('data/CMU_MOSEI/aligned/valid_text.npy')
    val_labels = np.load('data/CMU_MOSEI/aligned/valid_labels.npy')
    
    # Print data shapes
    logger.info(f"\nData shapes:")
    logger.info(f"Train text: {train_text.shape}")
    logger.info(f"Train labels: {train_labels.shape}")
    logger.info(f"Val text: {val_text.shape}")
    logger.info(f"Val labels: {val_labels.shape}")
    
    # Print label distribution
    logger.info("\nLabel distribution:")
    for split, labels in [("Train", train_labels), ("Val", val_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"\n{split} set:")
        for label, count in zip(unique, counts):
            logger.info(f"Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    # Normalize features
    logger.info("\nNormalizing training data...")
    train_text = normalize_features(train_text)
    logger.info("\nNormalizing validation data...")
    val_text = normalize_features(val_text)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_text),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_text),
        torch.LongTensor(val_labels)
    )
    
    # Create data loaders with weighted sampling
    class_weights = 1.0 / np.bincount(train_labels)
    sample_weights = class_weights[train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    # Initialize model
    model = TextTransformer(
        input_dim=train_text.shape[2],
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=5
    ).to(device)
    logger.info(f"\nModel architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer with class weights
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(NUM_EPOCHS):
        logger.info(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
        
        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        logger.info(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_text_model.pth')
            logger.info(f'New best model saved with validation F1: {val_f1:.4f}')

if __name__ == '__main__':
    main() 