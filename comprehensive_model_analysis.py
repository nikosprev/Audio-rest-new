#!/usr/bin/env python3
"""
Comprehensive Apollo Model Analysis and Comparison
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
from scipy import signal
from scipy.stats import pearsonr
import pandas as pd

# Import our model classes
from train_apollo_improved import ImprovedApolloModel
from train_apollo_balanced import BalancedApolloModel

class AudioQualityMetrics:
    """Comprehensive audio quality assessment"""
    
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
    
    def calculate_spectral_centroid(self, audio):
        """Spectral centroid"""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        return librosa.feature.spectral_centroid(S=np.abs(stft), sr=self.sr).mean()
    
    def calculate_spectral_rolloff(self, audio):
        """Spectral rolloff"""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        return librosa.feature.spectral_rolloff(S=np.abs(stft), sr=self.sr).mean()
    
    def calculate_zero_crossing_rate(self, audio):
        """Zero crossing rate"""
        return librosa.feature.zero_crossing_rate(audio).mean()
    
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
        
        # Spectral features
        metrics['spectral_centroid_clean'] = self.calculate_spectral_centroid(clean)
        metrics['spectral_centroid_degraded'] = self.calculate_spectral_centroid(degraded)
        metrics['spectral_centroid_restored'] = self.calculate_spectral_centroid(restored)
        
        metrics['spectral_rolloff_clean'] = self.calculate_spectral_rolloff(clean)
        metrics['spectral_rolloff_degraded'] = self.calculate_spectral_rolloff(degraded)
        metrics['spectral_rolloff_restored'] = self.calculate_spectral_rolloff(restored)
        
        metrics['zcr_clean'] = self.calculate_zero_crossing_rate(clean)
        metrics['zcr_degraded'] = self.calculate_zero_crossing_rate(degraded)
        metrics['zcr_restored'] = self.calculate_zero_crossing_rate(restored)
        
        return metrics

class ModelTester:
    """Test and compare different Apollo models"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.metrics = AudioQualityMetrics()
        self.results = {}
    
    def load_model(self, model_class, model_path, input_size):
        """Load a trained model"""
        model = model_class(input_size=input_size)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def test_model(self, model, test_data, model_name):
        """Test a single model"""
        print(f"\nTesting {model_name}...")
        
        all_metrics = []
        processing_times = []
        
        with torch.no_grad():
            for i, sample in enumerate(tqdm(test_data, desc=f"Testing {model_name}")):
                degraded = torch.FloatTensor(sample['degraded']).unsqueeze(0).to(self.device)
                clean = sample['clean']
                
                # Measure processing time
                start_time = time.time()
                restored = model(degraded)
                processing_time = time.time() - start_time
                
                restored = restored.cpu().numpy().flatten()
                
                # Calculate metrics
                metrics = self.metrics.calculate_all_metrics(clean, sample['degraded'], restored)
                metrics['processing_time'] = processing_time
                all_metrics.append(metrics)
                processing_times.append(processing_time)
        
        # Aggregate results
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        self.results[model_name] = {
            'individual_metrics': all_metrics,
            'aggregate_metrics': avg_metrics,
            'total_samples': len(test_data)
        }
        
        print(f"✓ {model_name} completed - Avg SNR: {avg_metrics['avg_snr']:.2f} dB")
    
    def run_comprehensive_test(self, h5_path, output_dir):
        """Run comprehensive testing on all available models"""
        
        # Load test data
        print("Loading test data...")
        with h5py.File(h5_path, 'r') as f:
            test_clean = f['val/clean'][:]
            test_degraded = f['val/degraded'][:]
        
        test_data = []
        for i in range(len(test_clean)):
            test_data.append({
                'clean': test_clean[i],
                'degraded': test_degraded[i]
            })
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Test different models
        models_to_test = []
        
        # Check for improved model
        improved_path = "apollo_improved/best_model.pth"
        if os.path.exists(improved_path):
            models_to_test.append({
                'name': 'Apollo Improved',
                'class': ImprovedApolloModel,
                'path': improved_path
            })
        
        # Check for balanced model (if trained)
        balanced_path = "apollo_balanced/best_model.pth"
        if os.path.exists(balanced_path):
            models_to_test.append({
                'name': 'Apollo Balanced',
                'class': BalancedApolloModel,
                'path': balanced_path
            })
        
        # Test each model
        for model_info in models_to_test:
            try:
                model = self.load_model(
                    model_info['class'], 
                    model_info['path'], 
                    input_size=test_data[0]['degraded'].size
                )
                self.test_model(model, test_data, model_info['name'])
            except Exception as e:
                print(f"Error testing {model_info['name']}: {e}")
        
        # Generate comprehensive report
        self.generate_report(output_dir)
    
    def generate_report(self, output_dir):
        """Generate comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            print("\n❌ No models were successfully tested. Please retrain a model and try again.")
            return
        
        # Create summary table
        summary_data = []
        for model_name, result in self.results.items():
            metrics = result['aggregate_metrics']
            summary_data.append({
                'Model': model_name,
                'Samples': result['total_samples'],
                'Avg SNR (dB)': f"{metrics['avg_snr']:.2f} ± {metrics['std_snr']:.2f}",
                'Avg PSNR (dB)': f"{metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f}",
                'Spectral Conv': f"{metrics['avg_spectral_convergence']:.4f} ± {metrics['std_spectral_convergence']:.4f}",
                'RMS Reduction (%)': f"{metrics['avg_rms_reduction']:.1f} ± {metrics['std_rms_reduction']:.1f}",
                'Avg Processing Time (s)': f"{metrics['avg_processing_time']:.3f} ± {metrics['std_processing_time']:.3f}"
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
        
        # Create detailed results
        detailed_results = {}
        for model_name, result in self.results.items():
            detailed_results[model_name] = {
                'aggregate_metrics': result['aggregate_metrics'],
                'total_samples': result['total_samples']
            }
        
        with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Generate visualizations
        self.create_visualizations(output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL ANALYSIS RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\nDetailed results saved to: {output_dir}")
    
    def create_visualizations(self, output_dir):
        """Create visualization plots"""
        if not self.results:
            print("No results to plot.")
            return
        plt.style.use('seaborn-v0_8')
        
        # SNR comparison
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: SNR comparison
        plt.subplot(2, 2, 1)
        snr_data = []
        model_names = []
        for model_name, result in self.results.items():
            snr_values = [m['snr'] for m in result['individual_metrics']]
            snr_data.append(snr_values)
            model_names.append(model_name)
        
        plt.boxplot(snr_data, labels=model_names)
        plt.title('SNR Comparison')
        plt.ylabel('SNR (dB)')
        plt.xticks(rotation=45)
        
        # Subplot 2: PSNR comparison
        plt.subplot(2, 2, 2)
        psnr_data = []
        for model_name, result in self.results.items():
            psnr_values = [m['psnr'] for m in result['individual_metrics']]
            psnr_data.append(psnr_values)
        
        plt.boxplot(psnr_data, labels=model_names)
        plt.title('PSNR Comparison')
        plt.ylabel('PSNR (dB)')
        plt.xticks(rotation=45)
        
        # Subplot 3: RMS reduction
        plt.subplot(2, 2, 3)
        rms_data = []
        for model_name, result in self.results.items():
            rms_values = [m['rms_reduction'] for m in result['individual_metrics']]
            rms_data.append(rms_values)
        
        plt.boxplot(rms_data, labels=model_names)
        plt.title('RMS Reduction')
        plt.ylabel('RMS Reduction (%)')
        plt.xticks(rotation=45)
        
        # Subplot 4: Processing time
        plt.subplot(2, 2, 4)
        time_data = []
        for model_name, result in self.results.items():
            time_values = [m['processing_time'] for m in result['individual_metrics']]
            time_data.append(time_values)
        
        plt.boxplot(time_data, labels=model_names)
        plt.title('Processing Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap for one model
        if self.results:
            model_name = list(self.results.keys())[0]
            result = self.results[model_name]
            
            # Select key metrics for correlation
            key_metrics = ['snr', 'psnr', 'spectral_convergence', 'rms_reduction', 
                          'dynamic_range_restored', 'spectral_centroid_restored']
            
            metric_data = []
            for sample_metrics in result['individual_metrics']:
                metric_data.append([sample_metrics[m] for m in key_metrics])
            
            correlation_matrix = np.corrcoef(np.array(metric_data).T)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, 
                       xticklabels=key_metrics, 
                       yticklabels=key_metrics, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       fmt='.2f')
            plt.title(f'Metric Correlations - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function to run comprehensive analysis"""
    
    # Setup
    device = 'cpu'  # Use CPU for consistent testing
    h5_path = "hdf5_datas/pmqd_dataset.h5"
    output_dir = "comprehensive_analysis_results"
    
    print("🎵 Apollo Model Comprehensive Analysis")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists(h5_path):
        print(f"❌ Dataset not found: {h5_path}")
        return
    
    # Create tester and run analysis
    tester = ModelTester(device=device)
    tester.run_comprehensive_test(h5_path, output_dir)
    
    print("\n✅ Analysis completed successfully!")
    print(f"📊 Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 