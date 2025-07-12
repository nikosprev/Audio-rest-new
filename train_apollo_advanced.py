#!/usr/bin/env python3
"""
Advanced Apollo training script with maximum complexity and attention mechanisms
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

class AttentionBlock(nn.Module):
    """Self-attention block for audio processing"""
    
    def __init__(self, channels, heads=8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        B, C, L = x.shape
        
        # Reshape for multi-head attention
        q = self.query(x).view(B, self.heads, self.head_dim, L)
        k = self.key(x).view(B, self.heads, self.head_dim, L)
        v = self.value(x).view(B, self.heads, self.head_dim, L)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.view(B, C, L)
        out = self.proj(out)
        
        return out + x  # Residual connection

class ResidualBlock(nn.Module):
    """Residual block with normalization and activation"""
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class AdvancedApolloModel(nn.Module):
    """Advanced Apollo-like model with attention mechanisms and maximum complexity"""
    
    def __init__(self, input_size=132300, hidden_size=1024):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initial feature extraction
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Encoder path (maximum complexity)
        self.encoder1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256, 11),
            ResidualBlock(256, 11),
            AttentionBlock(256, heads=8)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=9, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ResidualBlock(512, 9),
            ResidualBlock(512, 9),
            AttentionBlock(512, heads=8)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=7, padding=3),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            ResidualBlock(1024, 7),
            ResidualBlock(1024, 7),
            AttentionBlock(1024, heads=16)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv1d(1024, 2048, kernel_size=5, padding=2),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            ResidualBlock(2048, 5),
            ResidualBlock(2048, 5),
            AttentionBlock(2048, heads=16)
        )

        # Bottleneck with maximum capacity
        self.bottleneck = nn.Sequential(
            nn.Conv1d(2048, 4096, kernel_size=3, padding=1),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            ResidualBlock(4096, 3),
            ResidualBlock(4096, 3),
            ResidualBlock(4096, 3),
            AttentionBlock(4096, heads=32),
            nn.Conv1d(4096, 2048, kernel_size=3, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        # Decoder path (maximum complexity)
        self.decoder4 = nn.Sequential(
            nn.Conv1d(4096, 2048, kernel_size=5, padding=2),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            ResidualBlock(2048, 5),
            ResidualBlock(2048, 5),
            AttentionBlock(2048, heads=16)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv1d(3072, 1024, kernel_size=7, padding=3),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            ResidualBlock(1024, 7),
            ResidualBlock(1024, 7),
            AttentionBlock(1024, heads=16)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=9, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ResidualBlock(512, 9),
            ResidualBlock(512, 9),
            AttentionBlock(512, heads=8)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256, 11),
            ResidualBlock(256, 11),
            AttentionBlock(256, heads=8)
        )

        # Final output layers
        self.output_conv = nn.Sequential(
            nn.Conv1d(384, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=15, padding=7),
            nn.Tanh()
        )

        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x):
        batch_size = x.size(0)
        input_len = x.shape[1]
        x = x.view(batch_size, 1, -1)
        
        # Input processing
        x = self.input_conv(x)
        
        # Encoder path
        enc1 = self.encoder1(x)
        enc1_pooled = self.pool(enc1)
        
        enc2 = self.encoder2(enc1_pooled)
        enc2_pooled = self.pool(enc2)
        
        enc3 = self.encoder3(enc2_pooled)
        enc3_pooled = self.pool(enc3)
        
        enc4 = self.encoder4(enc3_pooled)
        enc4_pooled = self.pool(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pooled)
        
        # Decoder path with skip connections
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
        
        # Final output
        output = self.output_conv(dec1)
        output = output.view(batch_size, -1)
        
        # Ensure output length matches input
        if output.shape[1] > input_len:
            output = output[:, :input_len]
        elif output.shape[1] < input_len:
            pad = input_len - output.shape[1]
            output = torch.nn.functional.pad(output, (0, pad))
        
        return output

class AdvancedPerceptualLoss(nn.Module):
    """Advanced perceptual loss with multiple components"""
    
    def __init__(self, sr=44100):
        super().__init__()
        self.sr = sr
        
    def forward(self, pred, target):
        # MSE loss
        mse_loss = nn.MSELoss()(pred, target)
        
        # Spectral loss
        spec_loss = self.spectral_loss(pred, target)
        
        # Content loss
        content_loss = self.content_loss(pred, target)
        
        # Phase loss
        phase_loss = self.phase_loss(pred, target)
        
        # Envelope loss
        envelope_loss = self.envelope_loss(pred, target)
        
        # Combine losses with careful weighting
        total_loss = (
            mse_loss + 
            0.05 * spec_loss + 
            0.02 * content_loss + 
            0.01 * phase_loss + 
            0.03 * envelope_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'spectral': spec_loss.item(),
            'content': content_loss.item(),
            'phase': phase_loss.item(),
            'envelope': envelope_loss.item()
        }
    
    def spectral_loss(self, pred, target):
        """Enhanced spectral convergence loss"""
        # Compute STFT with multiple window sizes
        losses = []
        for n_fft in [1024, 2048, 4096]:
            pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
            target_stft = torch.stft(target, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
            
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            numerator = torch.norm(target_mag - pred_mag, p='fro')
            denominator = torch.norm(target_mag, p='fro')
            
            losses.append(numerator / (denominator + 1e-8))
        
        return torch.mean(torch.stack(losses))
    
    def content_loss(self, pred, target):
        """Enhanced content loss"""
        # Multiple scales for content preservation
        losses = []
        for kernel_size in [500, 1000, 2000]:
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            kernel = kernel.to(pred.device)
            
            pred_env = torch.abs(pred)
            target_env = torch.abs(target)
            
            pred_env_smooth = torch.nn.functional.conv1d(
                pred_env.unsqueeze(1), kernel, padding=kernel_size//2
            ).squeeze(1)
            target_env_smooth = torch.nn.functional.conv1d(
                target_env.unsqueeze(1), kernel, padding=kernel_size//2
            ).squeeze(1)
            
            losses.append(nn.MSELoss()(pred_env_smooth, target_env_smooth))
        
        return torch.mean(torch.stack(losses))
    
    def phase_loss(self, pred, target):
        """Phase coherence loss"""
        # Compute STFT
        pred_stft = torch.stft(pred, n_fft=2048, hop_length=512, return_complex=True)
        target_stft = torch.stft(target, n_fft=2048, hop_length=512, return_complex=True)
        
        # Phase difference
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        
        phase_diff = torch.cos(pred_phase - target_phase)
        return 1.0 - torch.mean(phase_diff)
    
    def envelope_loss(self, pred, target):
        """Envelope preservation loss"""
        # Compute Hilbert envelope
        def hilbert_envelope(x):
            # Simple envelope approximation
            return torch.sqrt(x**2 + torch.roll(x, 1, dims=-1)**2)
        
        pred_env = hilbert_envelope(pred)
        target_env = hilbert_envelope(target)
        
        return nn.MSELoss()(pred_env, target_env)

class AdvancedAudioRestorationTrainer:
    """Advanced trainer with sophisticated training strategies"""
    
    def __init__(self, model, device, lr=0.0001):
        self.model = model.to(device)
        self.device = device
        
        # Advanced optimizer with different learning rates for different layers
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'attention' in n], 'lr': lr * 2},
            {'params': [p for n, p in model.named_parameters() if 'attention' not in n], 'lr': lr}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        self.criterion = AdvancedPerceptualLoss()
        
        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=lr/100
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        loss_components = {'mse': 0, 'spectral': 0, 'content': 0, 'phase': 0, 'envelope': 0}
        
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
            
            # Advanced gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
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
        loss_components = {'mse': 0, 'spectral': 0, 'content': 0, 'phase': 0, 'envelope': 0}
        
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_components': loss_components
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def train_model(h5_path, output_dir, num_epochs=100, batch_size=2, lr=0.0001):
    """Train the advanced audio restoration model"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = PMQDDataset(h5_path, 'train')
    val_dataset = PMQDDataset(h5_path, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model with correct input size
    sample_data = train_dataset[0]
    input_size = sample_data['degraded'].numel()
    print(f"Input size: {input_size}")
    
    model = AdvancedApolloModel(input_size=input_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AdvancedAudioRestorationTrainer(model, device, lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 15
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting advanced training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_components = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_components = trainer.validate(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        trainer.scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"  MSE: {train_components['mse']:.6f}, Spec: {train_components['spectral']:.6f}")
        print(f"  Content: {train_components['content']:.6f}, Phase: {train_components['phase']:.6f}")
        print(f"  Envelope: {train_components['envelope']:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"  MSE: {val_components['mse']:.6f}, Spec: {val_components['spectral']:.6f}")
        print(f"  Content: {val_components['content']:.6f}, Phase: {val_components['phase']:.6f}")
        print(f"  Envelope: {val_components['envelope']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(output_dir, 'best_model.pth'),
                epoch, train_loss, val_loss, val_components
            )
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch, train_loss, val_loss, val_components
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1
    }
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nAdvanced training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final epoch: {epoch + 1}")
    print(f"Model saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train advanced Apollo model on PMQD dataset")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output_dir", type=str, default="./apollo_advanced", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    
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