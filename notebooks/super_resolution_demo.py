#!/usr/bin/env python
# coding: utf-8

# # Satellite Image Super-Resolution Demo
# 
# This script demonstrates how to use the super-resolution models to enhance satellite images.

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add parent directory to path
sys.path.append('..')

from models.esrgan import RRDBNet
from models.srcnn import SRCNN, FSRCNN
from utils.visualization import tensor_to_numpy, plot_comparison

# ## Load a pre-trained model
# 
# First, let's load a pre-trained super-resolution model. You can choose between ESRGAN, SRCNN, or FSRCNN.

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model parameters
model_type = 'esrgan'  # 'esrgan', 'srcnn', or 'fsrcnn'
scale_factor = 4

# Load model
if model_type == 'esrgan':
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        nf=64,
        nb=23,
        scale=scale_factor
    ).to(device)
    model_path = '../checkpoints/esrgan/netG_final.pth'
elif model_type == 'srcnn':
    model = SRCNN(num_channels=3).to(device)
    model_path = '../checkpoints/srcnn/model_final.pth'
elif model_type == 'fsrcnn':
    model = FSRCNN(scale_factor=scale_factor, num_channels=3).to(device)
    model_path = '../checkpoints/fsrcnn/model_final.pth'
else:
    raise ValueError(f"Unknown model type: {model_type}")

# Load weights if model file exists
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
else:
    print(f"Warning: Model file {model_path} not found. Using untrained model.")

# Set model to evaluation mode
model.eval()

# ## Load and preprocess an image
# 
# Now, let's load a satellite image and preprocess it for super-resolution.

def preprocess_image(image_path, scale_factor=None):
    """
    Preprocess image for inference
    
    Args:
        image_path (str): Path to the image
        scale_factor (int, optional): Scale factor for SRCNN
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Read image
    if image_path.endswith('.tif'):
        import rasterio
        with rasterio.open(image_path) as src:
            img = src.read()
            # Convert from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if scale_factor is not None:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img

# Load an image
image_path = '../data/raw/sample_image.jpg'  # Replace with your image path

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} not found. Please provide a valid image path.")
else:
    # Preprocess image
    lr_img = preprocess_image(image_path)
    
    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor_to_numpy(lr_img))
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

# ## Perform Super-Resolution
# 
# Now, let's enhance the resolution of the image using our model.

def super_resolve_image(model, lr_img, model_type='esrgan', scale_factor=4, device='cuda'):
    """
    Super-resolve an image
    
    Args:
        model (torch.nn.Module): Super-resolution model
        lr_img (torch.Tensor): Low-resolution image tensor
        model_type (str): Model type ('esrgan', 'srcnn', or 'fsrcnn')
        scale_factor (int): Scale factor for SRCNN
        device (str): Device to use
        
    Returns:
        torch.Tensor: Super-resolved image tensor
    """
    with torch.no_grad():
        lr_img = lr_img.to(device)
        
        if model_type == 'srcnn':
            # SRCNN expects the input to be already upscaled
            lr_upscaled = torch.nn.functional.interpolate(
                lr_img, 
                scale_factor=scale_factor, 
                mode='bicubic', 
                align_corners=False
            )
            sr_img = model(lr_upscaled)
        else:
            sr_img = model(lr_img)
    
    return sr_img

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} not found. Please provide a valid image path.")
else:
    # Super-resolve image
    sr_img = super_resolve_image(model, lr_img, model_type, scale_factor, device)
    
    # Create bicubic upscaled image for comparison
    bicubic_img = torch.nn.functional.interpolate(
        lr_img, 
        scale_factor=scale_factor, 
        mode='bicubic', 
        align_corners=False
    )
    
    # Convert to numpy for display
    lr_np = tensor_to_numpy(lr_img)
    sr_np = tensor_to_numpy(sr_img)
    bicubic_np = tensor_to_numpy(bicubic_img)
    
    # Display the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(lr_np)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(bicubic_np)
    plt.title("Bicubic Upscaling")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sr_np)
    plt.title(f"{model_type.upper()} Super-Resolution")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ## Compare Different Models
# 
# Let's compare the results of different super-resolution models.

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} not found. Please provide a valid image path.")
else:
    # Load models
    models = {}
    
    # ESRGAN
    esrgan_path = '../checkpoints/esrgan/netG_final.pth'
    if os.path.exists(esrgan_path):
        esrgan = RRDBNet(in_channels=3, out_channels=3, nf=64, nb=23, scale=scale_factor).to(device)
        esrgan.load_state_dict(torch.load(esrgan_path, map_location=device))
        esrgan.eval()
        models['esrgan'] = esrgan
    
    # SRCNN
    srcnn_path = '../checkpoints/srcnn/model_final.pth'
    if os.path.exists(srcnn_path):
        srcnn = SRCNN(num_channels=3).to(device)
        srcnn.load_state_dict(torch.load(srcnn_path, map_location=device))
        srcnn.eval()
        models['srcnn'] = srcnn
    
    # FSRCNN
    fsrcnn_path = '../checkpoints/fsrcnn/model_final.pth'
    if os.path.exists(fsrcnn_path):
        fsrcnn = FSRCNN(scale_factor=scale_factor, num_channels=3).to(device)
        fsrcnn.load_state_dict(torch.load(fsrcnn_path, map_location=device))
        fsrcnn.eval()
        models['fsrcnn'] = fsrcnn
    
    # If no models are available, use the current model
    if not models:
        models[model_type] = model
    
    # Super-resolve image with each model
    results = {}
    for name, model in models.items():
        results[name] = super_resolve_image(model, lr_img, name, scale_factor, device)
    
    # Create bicubic upscaled image for comparison
    bicubic_img = torch.nn.functional.interpolate(
        lr_img, 
        scale_factor=scale_factor, 
        mode='bicubic', 
        align_corners=False
    )
    results['bicubic'] = bicubic_img
    
    # Display the results
    plt.figure(figsize=(15, 5 * (len(results) // 2 + 1)))
    
    # Original image
    plt.subplot(len(results) // 2 + 1, 2, 1)
    plt.imshow(tensor_to_numpy(lr_img))
    plt.title("Original Image")
    plt.axis('off')
    
    # Results
    for i, (name, img) in enumerate(results.items(), 1):
        plt.subplot(len(results) // 2 + 1, 2, i + 1)
        plt.imshow(tensor_to_numpy(img))
        plt.title(f"{name.upper()} {'Upscaling' if name == 'bicubic' else 'Super-Resolution'}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ## Zoom in to Compare Details
# 
# Let's zoom in on a specific region to better compare the details.

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} not found. Please provide a valid image path.")
else:
    # Define the region to zoom in
    x1, y1, x2, y2 = 100, 100, 200, 200  # Replace with your coordinates
    
    # Extract the region from each image
    regions = {}
    
    # Original image
    lr_np = tensor_to_numpy(lr_img)
    lr_region = lr_np[y1//scale_factor:y2//scale_factor, x1//scale_factor:x2//scale_factor]
    regions['original'] = lr_region
    
    # Results
    for name, img in results.items():
        img_np = tensor_to_numpy(img)
        regions[name] = img_np[y1:y2, x1:x2]
    
    # Display the regions
    plt.figure(figsize=(15, 5 * (len(regions) // 3 + 1)))
    
    for i, (name, region) in enumerate(regions.items()):
        plt.subplot(len(regions) // 3 + 1, 3, i + 1)
        plt.imshow(region)
        plt.title(f"{name.upper()} {'Upscaling' if name == 'bicubic' else 'Super-Resolution'}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ## Save the Super-Resolved Image
# 
# Finally, let's save the super-resolved image to disk.

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Warning: Image file {image_path} not found. Please provide a valid image path.")
else:
    # Save the super-resolved image
    output_dir = '../results'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, img in results.items():
        if name == 'bicubic':
            continue
            
        # Convert to numpy
        img_np = tensor_to_numpy(img)
        
        # Convert to uint8
        img_np = (img_np * 255).astype(np.uint8)
        
        # Save image
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{name}.png")
        plt.imsave(output_path, img_np)
        print(f"Saved {name.upper()} super-resolved image to {output_path}") 