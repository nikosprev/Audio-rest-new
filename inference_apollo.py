#!/usr/bin/env python3
"""
Inference script for trained Apollo model
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
import json

class SimpleApolloModel(nn.Module):
    """Simplified Apollo-like model for audio restoration"""
    
    def __init__(self, input_size=132300, hidden_size=512, num_layers=6):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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
            nn.Linear(hidden_size // 2, input_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Reshape input to (batch, sequence_length, features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.hidden_size // 8)  # Reshape to sequence
        
        # Encode
        encoded = self.encoder(x.view(batch_size, -1))
        encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
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

def batch_process(input_dir, output_dir, model, device, sr=44100):
    """Process all audio files in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    audio_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3"))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Process audio
            restored_audio = process_audio_file(audio_file, model, device, sr)
            
            # Save restored audio
            output_file = output_path / f"restored_{audio_file.name}"
            sf.write(output_file, restored_audio, sr)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Run Apollo inference on audio files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for restored files")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument("--segment_length", type=int, default=3, help="Segment length in seconds")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, device = load_model(args.model_path)
    
    # Process audio files
    print(f"Processing audio files from {args.input_dir}")
    batch_process(args.input_dir, args.output_dir, model, device, args.sr)
    
    print(f"Processing complete! Restored files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 