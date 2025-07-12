<p align="center">
  <img src="asserts/apollo-logo.png" alt="Logo" width="150"/>
</p>

<p align="center">
  <strong>Kai Li<sup>1,2</sup>, Yi Luo<sup>2</sup></strong><br>
    <strong><sup>1</sup>Tsinghua University, Beijing, China</strong><br>
    <strong><sup>2</sup>Tencent AI Lab, Shenzhen, China</strong><br>
  <a href="https://arxiv.org/abs/2409.08514">ArXiv</a> | <a href="https://cslikai.cn/Apollo/">Demo</a>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Apollo" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/JusperLee/Apollo?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey">
</p>

<p align="center">

# Apollo Audio Restoration Model

This directory contains a simplified implementation of the Apollo audio restoration model, adapted for the PMQD (Perceived Music Quality Dataset) for audio restoration tasks.

## Setup

### Environment
- Python 3.10
- PyTorch 2.7.1
- Conda environment: `apollo`

### Installation
```bash
# Create conda environment
conda create -n apollo python=3.10 -y
conda activate apollo

# Install dependencies
pip install torch torchaudio torchvision torch-complex torch-mir-eval torch-optimizer torchmetrics pytorch-lightning pytorch-ranger
pip install librosa soundfile h5py hydra-core omegaconf pandas matplotlib
```

## Dataset Preparation

The PMQD dataset was prepared using the CSV mapping to properly pair clean and degraded audio files:

```bash
python prepare_pmqd_data_csv.py --csv_path ../pmqd.csv --clean_dir ../source_hq_music --degraded_dir ../degraded_dataset --output_dir ./hdf5_datas
```

This created:
- 788 matching pairs (630 training, 158 validation)
- HDF5 dataset with 3-second audio segments
- Apollo configuration file

## Training

The model was trained for 2 epochs and converged quickly:

```bash
python train_apollo.py --h5_path ./hdf5_datas/pmqd_dataset.h5 --output_dir ./apollo_training --epochs 10 --batch_size 2
```

**Training Results:**
- Best validation loss: 0.063940
- Training converged in 2 epochs
- Model architecture: Transformer-based encoder-decoder

## Model Architecture

The simplified Apollo model consists of:
- **Encoder**: 2-layer MLP (input_size → hidden_size)
- **Transformer**: 6-layer transformer encoder
- **Decoder**: 3-layer MLP (hidden_size → input_size)
- **Input size**: 132,300 samples (3 seconds at 44.1kHz)

## Testing Results

The model was tested on degraded audio files with the following results:

### Audio Quality Metrics
- **RMS Reduction**: 92-97% (very aggressive)
- **Peak Reduction**: 89-91%
- **Dynamic Range**: Improved by 1.6-11.5 dB
- **Audio Activity**: Reduced from 94-99% to 41.7%

### Issues Identified
1. **Over-processing**: The model is too aggressive in noise reduction
2. **Volume reduction**: Restored audio is very quiet (RMS reduced by 90%+)
3. **Loss of content**: Audio activity reduced significantly

## Files

- `prepare_pmqd_data_csv.py`: Dataset preparation using CSV mapping
- `train_apollo.py`: Training script for the Apollo model
- `test_apollo_model.py`: Testing script with quality metrics
- `compare_apollo_results.py`: Detailed analysis of restoration results
- `inference_apollo.py`: Inference script for new audio files
- `apollo_training/best_model.pth`: Trained model checkpoint
- `apollo_results/`: Restored audio files

## Next Steps

To improve the model performance:

1. **Adjust loss function**: Use perceptual loss instead of MSE
2. **Add regularization**: Prevent over-processing
3. **Fine-tune hyperparameters**: Learning rate, model size
4. **Use different architecture**: Consider U-Net or ResNet
5. **Data augmentation**: Add more training data
6. **Post-processing**: Normalize output levels

## Usage

```bash
# Train the model
python train_apollo.py --h5_path ./hdf5_datas/pmqd_dataset.h5 --output_dir ./apollo_training

# Test on audio files
python test_apollo_model.py --num_files 5

# Run inference on new files
python inference_apollo.py --model_path ./apollo_training/best_model.pth --input_dir <input_dir> --output_dir <output_dir>
```

## Notes

- The model shows potential but needs refinement to avoid over-processing
- Consider using the original Apollo architecture for better results
- The PMQD dataset provides good training data for audio restoration tasks
- Further experimentation with loss functions and model architectures is recommended
