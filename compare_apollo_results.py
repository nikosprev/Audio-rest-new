#!/usr/bin/env python3
"""
Compare Apollo restoration results
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_audio_file(file_path):
    """Analyze audio file characteristics"""
    audio, sr = librosa.load(file_path, sr=44100)
    
    # Basic stats
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    dynamic_range = 20 * np.log10(peak / (np.std(audio) + 1e-8))
    
    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
    zero_crossing = librosa.feature.zero_crossing_rate(audio).mean()
    
    return {
        'rms': rms,
        'peak': peak,
        'dynamic_range_db': dynamic_range,
        'spectral_centroid': spec_cent,
        'spectral_rolloff': spec_rolloff,
        'zero_crossing_rate': zero_crossing,
        'audio': audio
    }

def main():
    # Get test files
    degraded_dir = Path("../degraded_dataset")
    restored_dir = Path("./apollo_results")
    
    # Get first few files
    degraded_files = list(degraded_dir.glob("*.wav"))[:3]
    
    print("=== Apollo Model Analysis ===\n")
    
    for degraded_file in degraded_files:
        restored_file = restored_dir / f"apollo_restored_{degraded_file.name}"
        
        if not restored_file.exists():
            continue
            
        print(f"File: {degraded_file.name}")
        print("-" * 50)
        
        # Analyze both files
        degraded_stats = analyze_audio_file(degraded_file)
        restored_stats = analyze_audio_file(restored_file)
        
        # Print comparison
        print(f"RMS: {degraded_stats['rms']:.6f} -> {restored_stats['rms']:.6f} ({((restored_stats['rms'] - degraded_stats['rms']) / degraded_stats['rms'] * 100):+.1f}%)")
        print(f"Peak: {degraded_stats['peak']:.6f} -> {restored_stats['peak']:.6f} ({((restored_stats['peak'] - degraded_stats['peak']) / degraded_stats['peak'] * 100):+.1f}%)")
        print(f"Dynamic Range: {degraded_stats['dynamic_range_db']:.1f} dB -> {restored_stats['dynamic_range_db']:.1f} dB ({restored_stats['dynamic_range_db'] - degraded_stats['dynamic_range_db']:+.1f} dB)")
        print(f"Spectral Centroid: {degraded_stats['spectral_centroid']:.0f} Hz -> {restored_stats['spectral_centroid']:.0f} Hz ({((restored_stats['spectral_centroid'] - degraded_stats['spectral_centroid']) / degraded_stats['spectral_centroid'] * 100):+.1f}%)")
        print(f"Zero Crossing Rate: {degraded_stats['zero_crossing_rate']:.4f} -> {restored_stats['zero_crossing_rate']:.4f} ({((restored_stats['zero_crossing_rate'] - degraded_stats['zero_crossing_rate']) / degraded_stats['zero_crossing_rate'] * 100):+.1f}%)")
        
        # Check if audio is too quiet
        if restored_stats['rms'] < 0.001:
            print("⚠️  WARNING: Restored audio is very quiet!")
        
        # Check if audio is mostly silence
        silence_threshold = 0.01
        degraded_activity = np.sum(np.abs(degraded_stats['audio']) > silence_threshold) / len(degraded_stats['audio'])
        restored_activity = np.sum(np.abs(restored_stats['audio']) > silence_threshold) / len(restored_stats['audio'])
        
        print(f"Audio Activity: {degraded_activity:.1%} -> {restored_activity:.1%}")
        
        if restored_activity < 0.1:
            print("⚠️  WARNING: Restored audio has very low activity!")
        
        print()

if __name__ == "__main__":
    main() 