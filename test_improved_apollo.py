#!/usr/bin/env python3
"""
Test and compare improved Apollo model with original
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

# Import both model architectures
from train_apollo_improved import ImprovedApolloModel
from train_apollo import SimpleApolloModel

def load_improved_model(model_path, input_size=132300):
    """Load improved trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedApolloModel(input_size=input_size)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Improved model loaded successfully!")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Training epoch: {checkpoint['epoch']}")
    
    return model, device

def load_original_model(model_path, input_size=132300):
    """Load original trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleApolloModel(input_size=input_size)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Original model loaded successfully!")
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
    
    # Calculate dynamic range
    dynamic_range_original = 20 * np.log10(np.max(np.abs(original)) / (np.std(original) + 1e-8))
    dynamic_range_restored = 20 * np.log10(np.max(np.abs(restored)) / (np.std(restored) + 1e-8))
    
    return {
        'rms_original': rms_original,
        'rms_restored': rms_restored,
        'rms_reduction_percent': (rms_original - rms_restored) / rms_original * 100,
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_improvement': snr_after - snr_before,
        'spectral_centroid_original': spec_cent_original,
        'spectral_centroid_restored': spec_cent_restored,
        'dynamic_range_original': dynamic_range_original,
        'dynamic_range_restored': dynamic_range_restored,
        'dynamic_range_improvement': dynamic_range_restored - dynamic_range_original
    }

def main():
    parser = argparse.ArgumentParser(description="Compare improved vs original Apollo models")
    parser.add_argument("--original_model", type=str, default="./apollo_training/best_model.pth", help="Path to original model")
    parser.add_argument("--improved_model", type=str, default="./apollo_improved/best_model.pth", help="Path to improved model")
    parser.add_argument("--input_dir", type=str, default="../degraded_dataset", help="Input directory with audio files")
    parser.add_argument("--output_dir", type=str, default="./comparison_results", help="Output directory")
    parser.add_argument("--num_files", type=int, default=5, help="Number of files to test")
    
    args = parser.parse_args()
    
    # Load both models
    print("Loading models...")
    original_model, device = load_original_model(args.original_model)
    improved_model, _ = load_improved_model(args.improved_model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get test files
    input_path = Path(args.input_dir)
    audio_files = list(input_path.glob("*.wav"))[:args.num_files]
    
    print(f"Testing on {len(audio_files)} files...")
    
    original_results = []
    improved_results = []
    
    for audio_file in tqdm(audio_files, desc="Processing files"):
        try:
            # Load original audio for comparison
            original_audio, _ = librosa.load(audio_file, sr=44100)
            original_audio = original_audio / (np.max(np.abs(original_audio)) + 1e-8)
            
            # Process with original model
            original_restored = process_audio_file(audio_file, original_model, device)
            original_metrics = analyze_audio_quality(original_audio, original_restored)
            original_metrics['filename'] = audio_file.name
            original_results.append(original_metrics)
            
            # Process with improved model
            improved_restored = process_audio_file(audio_file, improved_model, device)
            improved_metrics = analyze_audio_quality(original_audio, improved_restored)
            improved_metrics['filename'] = audio_file.name
            improved_results.append(improved_metrics)
            
            # Save restored audio
            sf.write(Path(args.output_dir) / f"original_{audio_file.name}", original_restored, 44100)
            sf.write(Path(args.output_dir) / f"improved_{audio_file.name}", improved_restored, 44100)
            
            print(f"  {audio_file.name}:")
            print(f"    Original - RMS reduction: {original_metrics['rms_reduction_percent']:.1f}%, SNR: {original_metrics['snr_improvement']:.2f} dB")
            print(f"    Improved - RMS reduction: {improved_metrics['rms_reduction_percent']:.1f}%, SNR: {improved_metrics['snr_improvement']:.2f} dB")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Print comparison summary
    print(f"\n=== Model Comparison Results ===")
    print(f"Processed {len(original_results)} files")
    
    if original_results and improved_results:
        # Calculate averages
        avg_original_rms_reduction = np.mean([r['rms_reduction_percent'] for r in original_results])
        avg_improved_rms_reduction = np.mean([r['rms_reduction_percent'] for r in improved_results])
        avg_original_snr_improvement = np.mean([r['snr_improvement'] for r in original_results])
        avg_improved_snr_improvement = np.mean([r['snr_improvement'] for r in improved_results])
        avg_original_dynamic_improvement = np.mean([r['dynamic_range_improvement'] for r in original_results])
        avg_improved_dynamic_improvement = np.mean([r['dynamic_range_improvement'] for r in improved_results])
        
        print(f"\nOriginal Model:")
        print(f"  Average RMS reduction: {avg_original_rms_reduction:.1f}%")
        print(f"  Average SNR improvement: {avg_original_snr_improvement:.2f} dB")
        print(f"  Average dynamic range improvement: {avg_original_dynamic_improvement:.1f} dB")
        
        print(f"\nImproved Model:")
        print(f"  Average RMS reduction: {avg_improved_rms_reduction:.1f}%")
        print(f"  Average SNR improvement: {avg_improved_snr_improvement:.2f} dB")
        print(f"  Average dynamic range improvement: {avg_improved_dynamic_improvement:.1f} dB")
        
        print(f"\nImprovements:")
        print(f"  RMS reduction change: {avg_improved_rms_reduction - avg_original_rms_reduction:+.1f}%")
        print(f"  SNR improvement change: {avg_improved_snr_improvement - avg_original_snr_improvement:+.2f} dB")
        print(f"  Dynamic range improvement change: {avg_improved_dynamic_improvement - avg_original_dynamic_improvement:+.1f} dB")
        
        # Save detailed results
        comparison_data = {
            'original_results': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()} for r in original_results],
            'improved_results': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()} for r in improved_results],
            'summary': {
                'avg_original_rms_reduction': float(avg_original_rms_reduction),
                'avg_improved_rms_reduction': float(avg_improved_rms_reduction),
                'avg_original_snr_improvement': float(avg_original_snr_improvement),
                'avg_improved_snr_improvement': float(avg_improved_snr_improvement),
                'avg_original_dynamic_improvement': float(avg_original_dynamic_improvement),
                'avg_improved_dynamic_improvement': float(avg_improved_dynamic_improvement)
            }
        }
        
        with open(Path(args.output_dir) / 'comparison_results.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"To listen to results:")
    print(f"open {args.output_dir}")

if __name__ == "__main__":
    main() 