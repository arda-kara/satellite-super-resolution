# Satellite Image Super-Resolution (Work in Progress)

A deep learning-based solution for enhancing the resolution of satellite imagery, providing clearer and more detailed images for applications in remote sensing, terrain analysis, mapping, and 3D reconstruction.

> **Note:** This project is currently under active development. Some features may be incomplete or require further refinement.

## Current Status

The project currently has the following working components:

- ✅ ESRGAN model implementation for 4x super-resolution
- ✅ Command-line inference for processing individual images and directories
- ✅ Basic web application for uploading and enhancing images
- ✅ Color handling fixes to ensure accurate RGB/BGR conversions
- ✅ Debug logging throughout the pipeline for troubleshooting
- ✅ Unity integration utilities for game development

In progress:
- 🔄 Improving model training pipeline
- 🔄 Enhancing web application features
- 🔄 Optimizing performance for large images
- 🔄 Adding more comprehensive documentation

## Overview

This project implements state-of-the-art super-resolution techniques to upscale low-resolution satellite images while preserving and enhancing details. The system includes both a command-line interface for batch processing and a web application for interactive use.

## Features

- **Multiple Model Support**: ESRGAN (default), SRCNN, and FSRCNN implementations
- **4x Resolution Enhancement**: Transform low-resolution images into high-quality, detailed outputs
- **Web Interface**: User-friendly FastAPI application for uploading and enhancing images
- **Unity Integration**: Tools for integrating super-resolution into Unity-based applications
- **Consistent Color Handling**: Careful RGB/BGR conversion to maintain accurate colors
- **Debug Logging**: Comprehensive logging for troubleshooting and performance monitoring

## Project Structure

```
satellite-super-resolution/
├── app.py                   # FastAPI web application
├── checkpoints/             # Trained model weights
│   └── esrgan/              # ESRGAN model checkpoints
├── configs/                 # Configuration files
├── data/                    # Data storage
│   ├── raw/                 # Original satellite images
│   └── processed/           # Processed datasets
├── inference.py             # Inference script for command-line use
├── models/                  # Model architectures
│   ├── esrgan.py            # ESRGAN implementation
│   ├── srcnn.py             # SRCNN implementation
│   └── fsrcnn.py            # FSRCNN implementation
├── notebooks/               # Jupyter notebooks for experiments
├── requirements.txt         # Project dependencies
├── results/                 # Enhanced output images
├── static/                  # Static files for web app
├── templates/               # HTML templates for web app
├── train.py                 # Training script
├── unity_integration.py     # Unity integration utilities
└── utils/                   # Utility functions
    ├── data_preprocessing.py # Data preprocessing utilities
    ├── losses.py            # Loss functions
    ├── metrics.py           # Evaluation metrics
    └── visualization.py     # Visualization utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/satellite-super-resolution.git
cd satellite-super-resolution
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (optional):
```bash
# Models should be placed in the checkpoints directory
# ESRGAN model: checkpoints/esrgan/netG_final.pth
```

## Usage

### Command-line Interface

Process a single image:
```bash
python inference.py --model checkpoints/esrgan/netG_final.pth --model_type esrgan --input data/raw/sample_1.jpg --output enhanced_sample_1.jpg
```

Process a directory of images:
```bash
python inference.py --model checkpoints/esrgan/netG_final.pth --model_type esrgan --input data/raw/ --output results/
```

### Web Application

Start the web server:
```bash
uvicorn app:app --reload
```

Then open your browser and navigate to `http://127.0.0.1:8000` to use the web interface.

### Training

Train a new model:
```bash
python train.py --lr_dir data/processed/lr --hr_dir data/processed/hr --model_type esrgan --num_epochs 100
```

### Unity Integration

Process terrain images for Unity:
```bash
python unity_integration.py --input terrain_images --output enhanced_terrain --model checkpoints/esrgan/netG_final.pth --model_type esrgan
```
   Note: Pre-trained models are not included in this repository due to their size.
   You can ask for them via email and place them in the
   `checkpoints/esrgan/` directory, or train one yourself. I advise you to use CUDA if possible since train time is around 2.5 hours.


## Technical Details

### Model Architecture

The primary model is ESRGAN (Enhanced Super-Resolution Generative Adversarial Network), which consists of:
- A generator with Residual-in-Residual Dense Blocks (RRDB)
- A discriminator for adversarial training
- Perceptual loss using VGG features

### Color Handling

The project carefully manages color space conversions between RGB and BGR formats:
- OpenCV reads images in BGR format
- PyTorch and most deep learning models expect RGB format
- All conversions are explicitly handled to ensure color accuracy

### Known Issues

1. **Purple/Discolored Output Images**: We've implemented fixes for color channel mismatches, but some images may still show color artifacts depending on their format and source.

2. **Memory Usage**: Processing very large satellite images can require significant memory. We're working on optimizing this.

3. **Training Data**: The model performs best on the types of satellite imagery it was trained on. Results may vary with different satellite sources.

## Future Roadmap

- [ ] Support for additional super-resolution architectures
- [ ] Real-time processing capabilities
- [ ] Integration with cloud-based satellite imagery providers
- [ ] Specialized models for different types of satellite imagery
- [ ] Comprehensive documentation and tutorials
- [ ] Docker containerization for easy deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgements

- [ESRGAN Paper](https://arxiv.org/abs/1809.00219)
- [Sentinel-2 Dataset](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [SpaceNet Dataset](https://spacenet.ai/datasets/) 