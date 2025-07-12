#!/usr/bin/env python3
"""
Test script for trained Apollo model
"""

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def load_model(model_path, input_size=132300):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleApolloModel(input_size=input_size)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Training epoch: {checkpoint['epoch']}")
    
    return model, device

def process_audio_file(audio_path, model, device, sr=44100, segment_length=3):
    """Process a single audio file"""
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Process in segments
    segment_samples = sr * segment_length
    restored_segments = []
    
    print(f"Processing audio file: {audio_path}")
    print(f"Audio length: {len(audio)/sr:.2f} seconds")
    
    for i in range(0, len(audio) - segment_samples, segment_samples // 2):  # 50% overlap
        segment = audio[i:i + segment_samples]
        
        # Pad if necessary
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        
        # Convert to tensor
        segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)
        
        # Restore
        with torch.no_grad():
            restored_segment = model(segment_tensor)
        
        restored_segments.append(restored_segment.cpu().numpy().squeeze())
    
    # Combine segments (simple overlap-add)
    restored_audio = np.zeros_like(audio)
    weights = np.zeros_like(audio)
    
    for i, segment in enumerate(restored_segments):
        start_idx = i * segment_samples // 2
        end_idx = start_idx + len(segment)
        
        if end_idx > len(restored_audio):
            end_idx = len(restored_audio)
            segment = segment[:end_idx - start_idx]
        
        restored_audio[start_idx:end_idx] += segment
        weights[start_idx:end_idx] += 1
    
    # Normalize by weights
    weights[weights == 0] = 1
    restored_audio /= weights
    
    return restored_audio

def analyze_audio_quality(original, restored):
    """Analyze audio quality metrics"""
    
    # Calculate RMS
    rms_original = np.sqrt(np.mean(original**2))
    rms_restored = np.sqrt(np.mean(restored**2))
    
    # Calculate SNR improvement
    noise_estimate = original - restored
    snr_before = 10 * np.log10(np.sum(original**2) / np.sum(noise_estimate**2))
    snr_after = 10 * np.log10(np.sum(restored**2) / np.sum(noise_estimate**2))
    
    # Calculate spectral centroid (brightness)
    spec_cent_original = librosa.feature.spectral_centroid(y=original, sr=44100).mean()
    spec_cent_restored = librosa.feature.spectral_centroid(y=restored, sr=44100).mean()
    
    return {
        'rms_original': rms_original,
        'rms_restored': rms_restored,
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_improvement': snr_after - snr_before,
        'spectral_centroid_original': spec_cent_original,
        'spectral_centroid_restored': spec_cent_restored
    }

def main():
    parser = argparse.ArgumentParser(description="Test Apollo model on audio files")
    parser.add_argument("--model_path", type=str, default="./apollo_training/best_model.pth", help="Path to trained model")
    parser.add_argument("--input_dir", type=str, default="../degraded_dataset", help="Input directory with audio files")
    parser.add_argument("--output_dir", type=str, default="./apollo_results", help="Output directory for restored files")
    parser.add_argument("--num_files", type=int, default=5, help="Number of files to test")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, device = load_model(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get test files
    input_path = Path(args.input_dir)
    audio_files = list(input_path.glob("*.wav"))[:args.num_files]
    
    print(f"Testing on {len(audio_files)} files...")
    
    results = []
    
    for audio_file in tqdm(audio_files, desc="Processing files"):
        try:
            # Process audio
            restored_audio = process_audio_file(audio_file, model, device)
            
            # Load original for comparison
            original_audio, _ = librosa.load(audio_file, sr=44100)
            original_audio = original_audio / (np.max(np.abs(original_audio)) + 1e-8)
            
            # Analyze quality
            quality_metrics = analyze_audio_quality(original_audio, restored_audio)
            quality_metrics['filename'] = audio_file.name
            results.append(quality_metrics)
            
            # Save restored audio
            output_file = Path(args.output_dir) / f"apollo_restored_{audio_file.name}"
            sf.write(output_file, restored_audio, 44100)
            
            print(f"  {audio_file.name}: SNR improvement = {quality_metrics['snr_improvement']:.2f} dB")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Print summary
    print(f"\n=== Apollo Model Test Results ===")
    print(f"Processed {len(results)} files")
    
    if results:
        avg_snr_improvement = np.mean([r['snr_improvement'] for r in results])
        avg_rms_reduction = np.mean([(r['rms_original'] - r['rms_restored']) / r['rms_original'] * 100 for r in results])
        
        print(f"Average SNR improvement: {avg_snr_improvement:.2f} dB")
        print(f"Average RMS reduction: {avg_rms_reduction:.1f}%")
        print(f"Restored files saved to: {args.output_dir}")
    
    print(f"\nTo listen to results:")
    print(f"open {args.output_dir}")

if __name__ == "__main__":
    main() 