#!/usr/bin/env python3
"""
Quick Apollo Audio Restoration Demo
Tests basic audio processing and generates sample metrics
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import h5py
import json
import time
from pathlib import Path
from tqdm import tqdm

class SimpleAudioProcessor:
    """Simple audio processing for demo purposes"""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def add_noise(self, audio, noise_level=0.1):
        """Add synthetic noise to audio"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def add_reverb(self, audio, room_size=0.3):
        """Add simple reverb effect"""
        # Simple convolution-based reverb
        reverb_length = int(room_size * self.sr)
        reverb_tail = np.exp(-np.linspace(0, 5, reverb_length))
        reverb_tail = reverb_tail / np.sum(reverb_tail)
        
        # Apply reverb
        reverb_audio = np.convolve(audio, reverb_tail, mode='same')
        return audio + 0.3 * reverb_audio
    
    def degrade_audio(self, audio, noise_level=0.1, reverb_level=0.3):
        """Apply multiple degradations"""
        degraded = audio.copy()
        
        # Add noise
        if noise_level > 0:
            degraded = self.add_noise(degraded, noise_level)
        
        # Add reverb
        if reverb_level > 0:
            degraded = self.add_reverb(degraded, reverb_level)
        
        return degraded
    
    def simple_restoration(self, degraded_audio):
        """Simple restoration using basic signal processing"""
        original_length = len(degraded_audio)
        
        # Apply noise reduction using spectral subtraction
        stft = librosa.stft(degraded_audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.sr / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        cleaned_magnitude = magnitude - alpha * noise_spectrum
        cleaned_magnitude = np.maximum(cleaned_magnitude, beta * magnitude)
        
        # Reconstruct signal
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        restored = librosa.istft(cleaned_stft, hop_length=512, length=original_length)
        
        return restored

class AudioQualityMetrics:
    """Basic audio quality metrics"""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def calculate_snr(self, clean, restored):
        """Signal-to-Noise Ratio"""
        noise = clean - restored
        signal_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def calculate_psnr(self, clean, restored):
        """Peak Signal-to-Noise Ratio"""
        max_val = np.max(np.abs(clean))
        mse = np.mean((clean - restored)**2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    def calculate_spectral_convergence(self, clean, restored):
        """Spectral convergence loss"""
        clean_stft = librosa.stft(clean, n_fft=2048, hop_length=512)
        restored_stft = librosa.stft(restored, n_fft=2048, hop_length=512)
        
        clean_mag = np.abs(clean_stft)
        restored_mag = np.abs(restored_stft)
        
        numerator = np.linalg.norm(clean_mag - restored_mag, ord='fro')
        denominator = np.linalg.norm(clean_mag, ord='fro')
        
        return numerator / (denominator + 1e-8)
    
    def calculate_rms_energy(self, audio):
        """RMS energy"""
        return np.sqrt(np.mean(audio**2))
    
    def calculate_dynamic_range(self, audio):
        """Dynamic range"""
        return np.max(audio) - np.min(audio)
    
    def calculate_all_metrics(self, clean, degraded, restored):
        """Calculate all quality metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['snr'] = self.calculate_snr(clean, restored)
        metrics['psnr'] = self.calculate_psnr(clean, restored)
        metrics['spectral_convergence'] = self.calculate_spectral_convergence(clean, restored)
        
        # Energy metrics
        metrics['rms_clean'] = self.calculate_rms_energy(clean)
        metrics['rms_degraded'] = self.calculate_rms_energy(degraded)
        metrics['rms_restored'] = self.calculate_rms_energy(restored)
        metrics['rms_reduction'] = (metrics['rms_degraded'] - metrics['rms_restored']) / metrics['rms_degraded'] * 100
        
        # Dynamic range
        metrics['dynamic_range_clean'] = self.calculate_dynamic_range(clean)
        metrics['dynamic_range_degraded'] = self.calculate_dynamic_range(degraded)
        metrics['dynamic_range_restored'] = self.calculate_dynamic_range(restored)
        
        return metrics

def quick_test_with_pmqd():
    """Quick test using PMQD dataset samples"""
    print("🎵 Quick Apollo Audio Restoration Demo")
    print("="*50)
    
    # Check if dataset exists
    h5_path = "hdf5_datas/pmqd_dataset.h5"
    if not os.path.exists(h5_path):
        print(f"❌ Dataset not found: {h5_path}")
        return
    
    # Load a few test samples
    print("Loading test samples...")
    with h5py.File(h5_path, 'r') as f:
        test_clean = f['val/clean'][:5]  # Just 5 samples for quick test
        test_degraded = f['val/degraded'][:5]
    
    print(f"Loaded {len(test_clean)} test samples")
    
    # Initialize processors
    processor = SimpleAudioProcessor()
    metrics = AudioQualityMetrics()
    
    # Test results
    all_results = []
    
    print("\nProcessing samples...")
    for i in tqdm(range(len(test_clean)), desc="Processing"):
        clean = test_clean[i]
        degraded = test_degraded[i]
        
        # Apply simple restoration
        start_time = time.time()
        restored = processor.simple_restoration(degraded)
        processing_time = time.time() - start_time
        
        # Calculate metrics
        sample_metrics = metrics.calculate_all_metrics(clean, degraded, restored)
        sample_metrics['processing_time'] = processing_time
        sample_metrics['sample_id'] = i
        
        all_results.append(sample_metrics)
        
        # Save audio files for comparison
        output_dir = "quick_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        sf.write(os.path.join(output_dir, f"sample_{i}_clean.wav"), clean, 44100)
        sf.write(os.path.join(output_dir, f"sample_{i}_degraded.wav"), degraded, 44100)
        sf.write(os.path.join(output_dir, f"sample_{i}_restored.wav"), restored, 44100)
    
    # Calculate averages
    avg_metrics = {}
    for key in all_results[0].keys():
        if key != 'sample_id':
            values = [r[key] for r in all_results]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
    
    # Print results
    print("\n" + "="*60)
    print("QUICK TEST RESULTS")
    print("="*60)
    print(f"Tested {len(all_results)} samples")
    print(f"Average SNR: {avg_metrics['avg_snr']:.2f} ± {avg_metrics['std_snr']:.2f} dB")
    print(f"Average PSNR: {avg_metrics['avg_psnr']:.2f} ± {avg_metrics['std_psnr']:.2f} dB")
    print(f"Spectral Convergence: {avg_metrics['avg_spectral_convergence']:.4f} ± {avg_metrics['std_spectral_convergence']:.4f}")
    print(f"RMS Reduction: {avg_metrics['avg_rms_reduction']:.1f} ± {avg_metrics['std_rms_reduction']:.1f}%")
    print(f"Average Processing Time: {avg_metrics['avg_processing_time']:.3f} ± {avg_metrics['std_processing_time']:.3f} seconds")
    
    # Save detailed results
    results = {
        'individual_results': all_results,
        'average_metrics': avg_metrics,
        'test_info': {
            'total_samples': len(all_results),
            'sample_rate': 44100,
            'restoration_method': 'spectral_subtraction'
        }
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = convert_numpy_types(results)
    
    with open(os.path.join(output_dir, 'quick_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Quick test completed!")
    print(f"📁 Audio files saved to: {output_dir}")
    print(f"📊 Detailed results: {output_dir}/quick_test_results.json")
    
    # Show sample-by-sample results
    print("\nSample-by-sample results:")
    print("-" * 40)
    for result in all_results:
        print(f"Sample {result['sample_id']}: SNR={result['snr']:.2f}dB, "
              f"PSNR={result['psnr']:.2f}dB, RMS_red={result['rms_reduction']:.1f}%")

def quick_test_with_synthetic():
    """Quick test with synthetic audio"""
    print("🎵 Quick Apollo Audio Restoration Demo (Synthetic)")
    print("="*60)
    
    # Generate synthetic audio
    sr = 44100
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a complex signal (sine + harmonics + noise)
    clean = (0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
             0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
             0.2 * np.sin(2 * np.pi * 1320 * t))  # E6 note
    
    # Add some natural variation
    clean *= (1 + 0.1 * np.sin(2 * np.pi * 2 * t))  # Amplitude modulation
    
    # Normalize
    clean = clean / np.max(np.abs(clean)) * 0.8
    
    # Degrade the audio
    processor = SimpleAudioProcessor(sr)
    degraded = processor.degrade_audio(clean, noise_level=0.2, reverb_level=0.4)
    
    # Restore
    start_time = time.time()
    restored = processor.simple_restoration(degraded)
    processing_time = time.time() - start_time
    
    # Calculate metrics
    metrics = AudioQualityMetrics(sr)
    sample_metrics = metrics.calculate_all_metrics(clean, degraded, restored)
    sample_metrics['processing_time'] = processing_time
    
    # Save audio files
    output_dir = "quick_test_synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    sf.write(os.path.join(output_dir, "synthetic_clean.wav"), clean, sr)
    sf.write(os.path.join(output_dir, "synthetic_degraded.wav"), degraded, sr)
    sf.write(os.path.join(output_dir, "synthetic_restored.wav"), restored, sr)
    
    # Print results
    print("\n" + "="*60)
    print("SYNTHETIC AUDIO TEST RESULTS")
    print("="*60)
    print(f"SNR: {sample_metrics['snr']:.2f} dB")
    print(f"PSNR: {sample_metrics['psnr']:.2f} dB")
    print(f"Spectral Convergence: {sample_metrics['spectral_convergence']:.4f}")
    print(f"RMS Reduction: {sample_metrics['rms_reduction']:.1f}%")
    print(f"Processing Time: {sample_metrics['processing_time']:.3f} seconds")
    
    # Save results
    results = {
        'metrics': sample_metrics,
        'test_info': {
            'audio_type': 'synthetic',
            'duration': duration,
            'sample_rate': sr,
            'restoration_method': 'spectral_subtraction'
        }
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = convert_numpy_types(results)
    
    with open(os.path.join(output_dir, 'synthetic_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Synthetic test completed!")
    print(f"📁 Audio files saved to: {output_dir}")
    print(f"📊 Results: {output_dir}/synthetic_test_results.json")

def main():
    """Main function"""
    print("Choose test type:")
    print("1. PMQD dataset test (real audio)")
    print("2. Synthetic audio test")
    print("3. Both tests")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        quick_test_with_pmqd()
    elif choice == "2":
        quick_test_with_synthetic()
    elif choice == "3":
        quick_test_with_pmqd()
        print("\n" + "="*60)
        quick_test_with_synthetic()
    else:
        print("Invalid choice. Running PMQD test...")
        quick_test_with_pmqd()

if __name__ == "__main__":
    main() 