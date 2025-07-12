#!/usr/bin/env python3
"""
Prepare PMQD dataset for Apollo training
Organizes degraded and clean audio pairs into HDF5 format
"""

import os
import h5py
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm
import random

def create_hdf5_dataset(clean_dir, degraded_dir, output_dir, sr=44100, segment_length=3):
    """
    Create HDF5 dataset from PMQD audio files
    
    Args:
        clean_dir: Directory containing clean audio files
        degraded_dir: Directory containing degraded audio files  
        output_dir: Output directory for HDF5 files
        sr: Sample rate (default 44100)
        segment_length: Length of segments in seconds (default 3)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    clean_files = list(Path(clean_dir).glob("*.wav"))
    degraded_files = list(Path(degraded_dir).glob("*.wav"))
    
    print(f"Found {len(clean_files)} clean files and {len(degraded_files)} degraded files")
    
    # Create file mapping
    file_pairs = []
    for clean_file in clean_files:
        degraded_file = Path(degraded_dir) / clean_file.name
        if degraded_file.exists():
            file_pairs.append((clean_file, degraded_file))
    
    print(f"Found {len(file_pairs)} matching pairs")
    
    # Create HDF5 file
    h5_path = os.path.join(output_dir, "pmqd_dataset.h5")
    
    with h5py.File(h5_path, 'w') as h5_file:
        # Create groups
        train_group = h5_file.create_group('train')
        val_group = h5_file.create_group('val')
        
        # Split data 80/20
        random.shuffle(file_pairs)
        split_idx = int(0.8 * len(file_pairs))
        train_pairs = file_pairs[:split_idx]
        val_pairs = file_pairs[split_idx:]
        
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Validation pairs: {len(val_pairs)}")
        
        # Process training data
        process_file_pairs(train_pairs, train_group, sr, segment_length, "train")
        
        # Process validation data  
        process_file_pairs(val_pairs, val_group, sr, segment_length, "val")
    
    print(f"Dataset saved to {h5_path}")
    return h5_path

def process_file_pairs(file_pairs, h5_group, sr, segment_length, split_name):
    """Process file pairs and create HDF5 datasets"""
    
    segment_samples = sr * segment_length
    
    # Lists to store segments
    clean_segments = []
    degraded_segments = []
    
    print(f"Processing {split_name} data...")
    
    for clean_file, degraded_file in tqdm(file_pairs):
        try:
            # Load audio files
            clean_audio, _ = librosa.load(clean_file, sr=sr)
            degraded_audio, _ = librosa.load(degraded_file, sr=sr)
            
            # Ensure same length
            min_length = min(len(clean_audio), len(degraded_audio))
            clean_audio = clean_audio[:min_length]
            degraded_audio = degraded_audio[:min_length]
            
            # Create segments
            for i in range(0, min_length - segment_samples, segment_samples // 2):  # 50% overlap
                clean_seg = clean_audio[i:i + segment_samples]
                degraded_seg = degraded_audio[i:i + segment_samples]
                
                # Normalize
                clean_seg = clean_seg / (np.max(np.abs(clean_seg)) + 1e-8)
                degraded_seg = degraded_seg / (np.max(np.abs(degraded_seg)) + 1e-8)
                
                clean_segments.append(clean_seg)
                degraded_segments.append(degraded_seg)
                
        except Exception as e:
            print(f"Error processing {clean_file}: {e}")
            continue
    
    # Convert to numpy arrays
    clean_segments = np.array(clean_segments)
    degraded_segments = np.array(degraded_segments)
    
    print(f"{split_name}: {len(clean_segments)} segments")
    
    # Save to HDF5
    h5_group.create_dataset('clean', data=clean_segments, compression='gzip')
    h5_group.create_dataset('degraded', data=degraded_segments, compression='gzip')

def create_apollo_config(output_dir):
    """Create Apollo configuration for PMQD dataset"""
    
    config_content = f"""exp: 
  dir: ./Exps
  name: Apollo_PMQD

datas:
  _target_: look2hear.datas.MusdbMoisesdbDataModule
  train_dir: {output_dir}
  eval_dir: {output_dir}
  codec_type: mp3
  codec_options:
    bitrate: random
    compression: random
    complexity: random
    vbr: random
  sr: 44100
  segments: 3
  num_stems: 2  # Clean and degraded
  snr_range: [-10, 10]
  num_samples: 40000
  batch_size: 1
  num_workers: 4

model:
  _target_: look2hear.models.apollo.Apollo
  sr: 44100
  win: 20
  feature_dim: 256
  layer: 6

discriminator:
  _target_: look2hear.discriminators.frequencydis.MultiFrequencyDiscriminator
  nch: 2
  window: [32, 64, 128, 256, 512, 1024, 2048]

optimizer_g:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.01

optimizer_d:
  _target_: torch.optim.AdamW
  lr: 0.0001
  weight_decay: 0.01
  betas: [0.5, 0.99]

scheduler_g:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 2
  gamma: 0.98

scheduler_d:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 2
  gamma: 0.98

loss_g:
  _target_: look2hear.losses.gan_losses.MultiFrequencyGenLoss
  eps: 1e-8

loss_d:
  _target_: look2hear.losses.gan_losses.MultiFrequencyDisLoss
  eps: 1e-8

metrics:
  _target_: look2hear.losses.MultiSrcNegSDR
  sdr_type: sisdr

system:
  _target_: look2hear.system.audio_litmodule.AudioLightningModule

early_stopping:
  _target_: pytorch_lightning.callbacks.EarlyStopping
  monitor: val_loss
  patience: 20
  mode: min
  verbose: true

checkpoint:
  _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: ${{exp.dir}}/${{exp.name}}/checkpoints
  monitor: val_loss
  mode: min
  verbose: true
  save_top_k: 5
  save_last: true
  filename: '{{epoch}}-{{val_loss:.4f}}'

logger:
  _target_: pytorch_lightning.loggers.WandbLogger
  name: ${{exp.name}}
  save_dir: ${{exp.dir}}/${{exp.name}}/logs
  offline: true
  project: Audio-Restoration

trainer:
  _target_: pytorch_lightning.Trainer
  devices: 1  # Single GPU for Mac
  max_epochs: 100
  sync_batchnorm: false
  default_root_dir: ${{exp.dir}}/${{exp.name}}/
  accelerator: auto
  limit_train_batches: 1.0
  fast_dev_run: false
"""
    
    config_path = os.path.join(output_dir, "apollo_pmqd.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Apollo config saved to {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Prepare PMQD dataset for Apollo")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory with clean audio files")
    parser.add_argument("--degraded_dir", type=str, required=True, help="Directory with degraded audio files")
    parser.add_argument("--output_dir", type=str, default="./hdf5_datas", help="Output directory for HDF5 files")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument("--segment_length", type=int, default=3, help="Segment length in seconds")
    
    args = parser.parse_args()
    
    # Create HDF5 dataset
    h5_path = create_hdf5_dataset(
        args.clean_dir, 
        args.degraded_dir, 
        args.output_dir, 
        args.sr, 
        args.segment_length
    )
    
    # Create Apollo config
    config_path = create_apollo_config(args.output_dir)
    
    print("\nDataset preparation complete!")
    print(f"HDF5 dataset: {h5_path}")
    print(f"Apollo config: {config_path}")
    print("\nTo train Apollo:")
    print(f"python train.py --config {config_path}")

if __name__ == "__main__":
    main() 