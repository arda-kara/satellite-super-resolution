# Satellite Super-Resolution Notebooks

This directory contains notebooks and scripts for demonstrating the satellite image super-resolution models.

## Contents

- `super_resolution_demo.py`: A Python script that demonstrates how to use the super-resolution models to enhance satellite images.

## Usage

To run the demo script:

```bash
python super_resolution_demo.py
```

Note: Before running the script, make sure you have:
1. Installed all the required dependencies (see the main README.md)
2. Either trained a model or downloaded a pre-trained model
3. Placed a sample satellite image in the `data/raw` directory

## Expected Output

The script will:
1. Load a pre-trained super-resolution model (ESRGAN, SRCNN, or FSRCNN)
2. Load and preprocess a sample satellite image
3. Enhance the resolution of the image using the model
4. Compare the results with bicubic upscaling
5. Compare different super-resolution models (if available)
6. Zoom in on specific regions to better compare the details
7. Save the super-resolved images to the `results` directory

## Customization

You can customize the script by modifying the following parameters:
- `model_type`: Choose between 'esrgan', 'srcnn', or 'fsrcnn'
- `scale_factor`: Set the upscaling factor (default: 4)
- `image_path`: Path to the input satellite image
- `x1, y1, x2, y2`: Coordinates for the zoom-in region 