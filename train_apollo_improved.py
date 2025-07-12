#!/usr/bin/env python3
"""
Improved Apollo training script with better loss functions and regularization
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

class PMQDDataset(Dataset):
    """Dataset for PMQD audio restoration"""
    
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        
        with h5py.File(h5_path, 'r') as f:
            self.clean_data = f[f'{split}/clean'][:]
            self.degraded_data = f[f'{split}/degraded'][:]
        
        print(f"Loaded {len(self.clean_data)} {split} samples")
    
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, idx):
        clean = torch.FloatTensor(self.clean_data[idx])
        degraded = torch.FloatTensor(self.degraded_data[idx])
        
        return {
            'clean': clean,
            'degraded': degraded
        }

class ImprovedApolloModel(nn.Module):
    """Improved Apollo-like model with deeper U-Net architecture"""
    
    def __init__(self, input_size=132300, hidden_size=512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_dim = int(np.sqrt(input_size))
        self.feature_dim = input_size // self.time_dim

        # Encoder path (deeper, more channels)
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        # Decoder path (deeper, more channels)
        self.decoder4 = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv1d(512 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x):
        batch_size = x.size(0)
        input_len = x.shape[1]
        x = x.view(batch_size, 1, -1)
        enc1 = self.encoder1(x)
        enc1_pooled = self.pool(enc1)
        enc2 = self.encoder2(enc1_pooled)
        enc2_pooled = self.pool(enc2)
        enc3 = self.encoder3(enc2_pooled)
        enc3_pooled = self.pool(enc3)
        enc4 = self.encoder4(enc3_pooled)
        enc4_pooled = self.pool(enc4)
        bottleneck = self.bottleneck(enc4_pooled)
        def crop_to_match(a, b):
            min_len = min(a.shape[2], b.shape[2])
            return a[:, :, :min_len], b[:, :, :min_len]
        dec4 = self.upsample(bottleneck)
        dec4, enc4 = crop_to_match(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upsample(dec4)
        dec3, enc3 = crop_to_match(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upsample(dec3)
        dec2, enc2 = crop_to_match(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upsample(dec2)
        dec1, enc1 = crop_to_match(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        output = dec1.view(batch_size, -1)
        if output.shape[1] > input_len:
            output = output[:, :input_len]
        elif output.shape[1] < input_len:
            pad = input_len - output.shape[1]
            output = torch.nn.functional.pad(output, (0, pad))
        return output

class PerceptualLoss(nn.Module):
    """Perceptual loss using spectral features"""
    
    def __init__(self, sr=44100):
        super().__init__()
        self.sr = sr
        
    def forward(self, pred, target):
        # MSE loss
        mse_loss = nn.MSELoss()(pred, target)
        
        # Spectral loss
        spec_loss = self.spectral_loss(pred, target)
        
        # Content loss (preserve overall structure)
        content_loss = self.content_loss(pred, target)
        
        # Combine losses
        total_loss = mse_loss + 0.1 * spec_loss + 0.05 * content_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'spectral': spec_loss.item(),
            'content': content_loss.item()
        }
    
    def spectral_loss(self, pred, target):
        """Spectral convergence loss"""
        # Compute STFT
        pred_stft = torch.stft(pred, n_fft=2048, hop_length=512, return_complex=True)
        target_stft = torch.stft(target, n_fft=2048, hop_length=512, return_complex=True)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # Spectral convergence
        numerator = torch.norm(target_mag - pred_mag, p='fro')
        denominator = torch.norm(target_mag, p='fro')
        
        return numerator / (denominator + 1e-8)
    
    def content_loss(self, pred, target):
        """Content loss to preserve overall structure"""
        # Compute envelope
        pred_env = torch.abs(pred)
        target_env = torch.abs(target)
        
        # Smooth envelope
        kernel_size = 1000
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        kernel = kernel.to(pred.device)
        
        pred_env_smooth = torch.nn.functional.conv1d(
            pred_env.unsqueeze(1), kernel, padding=kernel_size//2
        ).squeeze(1)
        target_env_smooth = torch.nn.functional.conv1d(
            target_env.unsqueeze(1), kernel, padding=kernel_size//2
        ).squeeze(1)
        
        return nn.MSELoss()(pred_env_smooth, target_env_smooth)

class AudioRestorationTrainer:
    """Improved trainer for audio restoration model"""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = PerceptualLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        loss_components = {'mse': 0, 'spectral': 0, 'content': 0}
        
        for batch in tqdm(dataloader, desc="Training"):
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            restored = self.model(degraded)
            
            # Calculate loss
            loss, components = self.criterion(restored, clean)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]
        
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        return avg_loss, loss_components
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        loss_components = {'mse': 0, 'spectral': 0, 'content': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                restored = self.model(degraded)
                loss, components = self.criterion(restored, clean)
                
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += components[key]
        
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        return avg_loss, loss_components
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss, loss_components):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_components': loss_components
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def train_model(h5_path, output_dir, num_epochs=30, batch_size=4, lr=0.0005):
    """Train the improved audio restoration model"""
    
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
    
    model = ImprovedApolloModel(input_size=input_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        train_loss, train_components = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_components = trainer.validate(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f} (MSE: {train_components['mse']:.6f}, Spec: {train_components['spectral']:.6f}, Content: {train_components['content']:.6f})")
        print(f"Val Loss: {val_loss:.6f} (MSE: {val_components['mse']:.6f}, Spec: {val_components['spectral']:.6f}, Content: {val_components['content']:.6f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(output_dir, 'best_model.pth'),
                epoch, train_loss, val_loss, val_components
            )
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch, train_loss, val_loss, val_components
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
    parser = argparse.ArgumentParser(description="Train improved Apollo model on PMQD dataset")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output_dir", type=str, default="./apollo_improved", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    
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