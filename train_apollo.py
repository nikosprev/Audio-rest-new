#!/usr/bin/env python3
"""
Simplified Apollo training script for PMQD dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import glob

class PMQDDataset(Dataset):
    """Dataset for PMQD audio restoration"""
    
    def __init__(self, h5_path, split='train'):
        self.split = split
        
        if os.path.isdir(h5_path):
            self.is_h5 = False
            self.data_dir = h5_path
            clean_dir = os.path.join(h5_path, 'small_wav')
            degraded_dir = os.path.join(h5_path, 'small_degraded_wav')
            
            if not os.path.exists(clean_dir) or not os.path.exists(degraded_dir):
                raise ValueError("Required subdirectories small_wav and small_degraded_wav not found")
            
            all_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
            
            if len(all_files) == 0:
                raise ValueError("No wav files found in small_wav")
            
            # Check degraded has same files
            degraded_files_set = set(os.listdir(degraded_dir))
            for f in all_files:
                if f not in degraded_files_set:
                    raise ValueError(f"Missing degraded file for {f}")
            
            # Split 80/20
            num_total = len(all_files)
            num_train = int(num_total * 0.8)
            if num_train == 0 and num_total > 0:
                num_train = num_total - 1  # Ensure at least one for val if small
            
            if self.split == 'train':
                self.files = all_files[:num_train]
            elif self.split == 'val':
                self.files = all_files[num_train:]
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            self.clean_files = [os.path.join(clean_dir, f) for f in self.files]
            self.degraded_files = [os.path.join(degraded_dir, f) for f in self.files]
            
            print(f"Loaded {len(self.files)} {split} file pairs from directory")
        else:
            self.is_h5 = True
            self.h5_path = h5_path
            with h5py.File(h5_path, 'r') as f:
                self.clean_data = f[f'{split}/clean'][:]
                self.degraded_data = f[f'{split}/degraded'][:]
            
            print(f"Loaded {len(self.clean_data)} {split} samples from HDF5")
    
    def __len__(self):
        if self.is_h5:
            return len(self.clean_data)
        else:
            return len(self.files)
    
    def __getitem__(self, idx):
        if self.is_h5:
            clean = torch.FloatTensor(self.clean_data[idx])
            degraded = torch.FloatTensor(self.degraded_data[idx])
        else:
            target_sr = 44100
            target_length = target_sr * 3  # 3 seconds
            
            clean_audio, _ = librosa.load(self.clean_files[idx], sr=target_sr)
            degraded_audio, _ = librosa.load(self.degraded_files[idx], sr=target_sr)
            
            # Truncate or pad to target length
            if len(clean_audio) > target_length:
                clean_audio = clean_audio[:target_length]
            else:
                clean_audio = np.pad(clean_audio, (0, target_length - len(clean_audio)), mode='constant')
            
            if len(degraded_audio) > target_length:
                degraded_audio = degraded_audio[:target_length]
            else:
                degraded_audio = np.pad(degraded_audio, (0, target_length - len(degraded_audio)), mode='constant')
            
            clean = torch.FloatTensor(clean_audio)
            degraded = torch.FloatTensor(degraded_audio)
        
        return {
            'clean': clean,
            'degraded': degraded
        }

class SimpleApolloModel(nn.Module):
    """Simplified Apollo-like model for audio restoration"""
    
    def __init__(self, input_size=132300, hidden_size=512, num_layers=6):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Encoder - process the full audio segment
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, input_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Flatten input
        x = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.unsqueeze(1)  # Add sequence dimension (batch, 1, hidden_size)
        
        # Transform
        transformed = self.transformer(encoded)
        
        # Decode
        decoded = self.decoder(transformed.squeeze(1))
        
        return decoded

class AudioRestorationTrainer:
    """Trainer for audio restoration model"""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            restored = self.model(degraded)
            
            # Calculate loss
            loss = self.criterion(restored, clean)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                restored = self.model(degraded)
                loss = self.criterion(restored, clean)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def train_model(h5_path, output_dir, num_epochs=50, batch_size=4, lr=0.001):
    """Train the audio restoration model"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = PMQDDataset(h5_path, 'train')
    val_dataset = PMQDDataset(h5_path, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model with correct input size
    sample_data = train_dataset[0]
    input_size = sample_data['degraded'].numel()
    print(f"Input size: {input_size}")
    
    model = SimpleApolloModel(input_size=input_size)
    
    # Create trainer
    trainer = AudioRestorationTrainer(model, device, lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(output_dir, 'best_model.pth'),
                epoch, train_loss, val_loss
            )
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch, train_loss, val_loss
            )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train Apollo model on PMQD dataset")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to HDF5 dataset or directory containing small_wav/small_degraded_wav")
    parser.add_argument("--output_dir", type=str, default="./apollo_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(
        args.h5_path,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.lr
    )

if __name__ == "__main__":
    main() 