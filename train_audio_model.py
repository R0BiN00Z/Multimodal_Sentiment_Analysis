import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from models.audio_model import AudioNet
from sklearn.metrics import accuracy_score, f1_score
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

def load_data():
    # Load training data
    train_audio = np.load('data/CMU_MOSEI/aligned/train_audio.npy')
    train_labels = np.load('data/CMU_MOSEI/aligned/train_labels.npy')
    
    # Load validation data
    val_audio = np.load('data/CMU_MOSEI/aligned/valid_audio.npy')
    val_labels = np.load('data/CMU_MOSEI/aligned/valid_labels.npy')
    
    # Convert to tensors
    train_audio = torch.FloatTensor(train_audio)
    train_labels = torch.LongTensor(train_labels)
    val_audio = torch.FloatTensor(val_audio)
    val_labels = torch.LongTensor(val_labels)
    
    # Create datasets
    train_dataset = TensorDataset(train_audio, train_labels)
    val_dataset = TensorDataset(val_audio, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 50 == 0:
            logging.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
    
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
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load data
    train_loader, val_loader = load_data()
    
    # Initialize model
    model = AudioNet().to(device)
    
    # Loss function with class weights
    class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(NUM_EPOCHS):
        logging.info(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        logging.info(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
        
        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        logging.info(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f'New best model saved with validation F1: {val_f1:.4f}')

if __name__ == '__main__':
    main() 