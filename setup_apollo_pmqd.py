#!/usr/bin/env python3
"""
Setup script for Apollo with PMQD dataset
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Setting up Apollo for PMQD dataset...")
    
    # Check if PMQD data exists
    pmqd_clean = "../source_hq_music"
    pmqd_degraded = "../degraded_dataset"
    
    if not os.path.exists(pmqd_clean):
        print(f"Error: Clean audio directory not found at {pmqd_clean}")
        print("Please ensure PMQD dataset is set up correctly")
        return
    
    if not os.path.exists(pmqd_degraded):
        print(f"Error: Degraded audio directory not found at {pmqd_degraded}")
        print("Please ensure PMQD dataset is set up correctly")
        return
    
    print(f"Found PMQD data:")
    print(f"  Clean: {pmqd_clean}")
    print(f"  Degraded: {pmqd_degraded}")
    
    # Create output directory
    output_dir = "./hdf5_datas"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run data preparation
    print("\nPreparing HDF5 dataset...")
    cmd = [
        "python", "prepare_pmqd_data.py",
        "--clean_dir", pmqd_clean,
        "--degraded_dir", pmqd_degraded,
        "--output_dir", output_dir,
        "--sr", "44100",
        "--segment_length", "3"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Data preparation completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during data preparation: {e}")
        print(f"Error output: {e.stderr}")
        return
    
    # Create training output directory
    training_output = "./apollo_training"
    os.makedirs(training_output, exist_ok=True)
    
    print(f"\nSetup complete!")
    print(f"HDF5 dataset: {output_dir}/pmqd_dataset.h5")
    print(f"Apollo config: {output_dir}/apollo_pmqd.yaml")
    print(f"Training output: {training_output}")
    
    print("\nTo start training, run:")
    print(f"python train_apollo.py --h5_path {output_dir}/pmqd_dataset.h5 --output_dir {training_output}")
    
    print("\nTo run inference after training:")
    print(f"python inference_apollo.py --model_path {training_output}/best_model.pth --input_dir <input_audio_dir> --output_dir <output_dir>")

if __name__ == "__main__":
    main() 