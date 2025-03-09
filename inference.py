import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import time
from PIL import Image
import rasterio
from rasterio.enums import Resampling
import json
from tqdm import tqdm

from models.esrgan import RRDBNet
from models.srcnn import SRCNN, FSRCNN
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import save_image, plot_comparison

# Debug flag
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

def preprocess_image(image_path, scale_factor=None):
    """
    Preprocess image for super-resolution
    
    Args:
        image_path (str): Path to the input image
        scale_factor (int, optional): Scale factor for resizing
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, C, H, W)
    """
    debug_print(f"Preprocessing image: {image_path}")
    
    # Read image
    if image_path.endswith('.tif'):
        debug_print("Reading TIFF image")
        with rasterio.open(image_path) as src:
            img = src.read()
            # Convert from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            debug_print(f"TIFF image shape: {img.shape}")
            
            # If single channel, convert to RGB
            if img.shape[2] == 1:
                debug_print("Converting single-channel image to RGB")
                img = np.repeat(img, 3, axis=2)
    else:
        debug_print("Reading regular image with OpenCV")
        # OpenCV reads in BGR, convert to RGB
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        debug_print(f"Original image shape: {img.shape}, converting BGR to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Handle RGBA images (4 channels) by removing alpha channel
    if img.shape[2] == 4:
        debug_print("Converting RGBA to RGB by removing alpha channel")
        img = img[:, :, :3]
    
    # Resize if scale_factor is provided
    if scale_factor is not None:
        debug_print(f"Resizing image by scale factor: {scale_factor}")
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    debug_print("Normalizing image to [0, 1]")
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    debug_print("Converting image to tensor")
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    debug_print(f"Preprocessed tensor shape: {img.shape}")
    return img

def super_resolve_image(model, lr_img, model_type='esrgan', scale_factor=4, device='cuda'):
    """
    Apply super-resolution to an image
    
    Args:
        model: The super-resolution model
        lr_img (torch.Tensor): Low-resolution image tensor of shape (1, C, H, W)
        model_type (str): Type of model ('esrgan', 'srcnn', or 'fsrcnn')
        scale_factor (int): Scale factor for super-resolution
        device (str): Device to use for inference
        
    Returns:
        torch.Tensor: Super-resolved image tensor of shape (1, C, H*scale, W*scale)
    """
    debug_print(f"Running super-resolution with model type: {model_type}")
    debug_print(f"Input tensor shape: {lr_img.shape}")
    
    # Move to device
    lr_img = lr_img.to(device)
    
    # Apply super-resolution
    with torch.no_grad():
        start_time = time.time()
        
        if model_type in ['srcnn', 'fsrcnn']:
            # For SRCNN/FSRCNN, we need to upscale the image first
            debug_print("Upscaling image for SRCNN/FSRCNN")
            h, w = lr_img.shape[2], lr_img.shape[3]
            upscaled_img = torch.nn.functional.interpolate(
                lr_img, size=(h * scale_factor, w * scale_factor), 
                mode='bicubic', align_corners=False
            )
            sr_img = model(upscaled_img)
        else:
            # For ESRGAN, the model handles upscaling
            sr_img = model(lr_img)
        
        elapsed = time.time() - start_time
    
    debug_print(f"Super-resolution completed in {elapsed:.2f} seconds")
    debug_print(f"Output tensor shape: {sr_img.shape}")
    
    return sr_img

def process_directory(model, input_dir, output_dir, model_type='esrgan', scale_factor=4, device='cuda'):
    """
    Process all images in a directory
    
    Args:
        model: The super-resolution model
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save output images
        model_type (str): Type of model ('esrgan', 'srcnn', or 'fsrcnn')
        scale_factor (int): Scale factor for super-resolution
        device (str): Device to use for inference
        
    Returns:
        dict: Dictionary with processing information
    """
    debug_print(f"Processing directory: {input_dir}")
    debug_print(f"Saving results to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    debug_print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = {}
    
    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"sr_{img_file}")
        
        debug_print(f"Processing image: {input_path}")
        
        # Preprocess image
        lr_img = preprocess_image(input_path)
        
        # Apply super-resolution
        start_time = time.time()
        sr_img = super_resolve_image(model, lr_img, model_type, scale_factor, device)
        elapsed = time.time() - start_time
        
        # Save output image
        debug_print(f"Saving super-resolved image to: {output_path}")
        save_image(sr_img[0], output_path)
        
        # Store results
        results[img_file] = {
            'input_path': input_path,
            'output_path': output_path,
            'processing_time': elapsed,
            'scale_factor': scale_factor
        }
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'processing_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    debug_print(f"Processed {len(image_files)} images")
    debug_print(f"Results saved to: {results_path}")
    
    return results

def load_model(model_path, model_type='esrgan', scale_factor=4, device='cuda'):
    """
    Load a super-resolution model
    
    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model ('esrgan', 'srcnn', or 'fsrcnn')
        scale_factor (int): Scale factor for super-resolution
        device (str): Device to use for inference
        
    Returns:
        model: The loaded model
    """
    debug_print(f"Loading {model_type} model from: {model_path}")
    
    # Create model
    if model_type == 'esrgan':
        model = RRDBNet(3, 3, 64, 23, gc=32)
    elif model_type == 'srcnn':
        model = SRCNN()
    elif model_type == 'fsrcnn':
        model = FSRCNN(scale_factor=scale_factor)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if os.path.exists(model_path):
        debug_print("Loading model weights")
        if device == 'cuda' and torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            debug_print("Loading from training checkpoint")
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            debug_print("Loading from state dict")
            model.load_state_dict(state_dict)
    else:
        debug_print(f"WARNING: Model file {model_path} not found. Using untrained model.")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Super-resolution inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="esrgan", choices=["esrgan", "srcnn", "fsrcnn"], 
                        help="Model architecture")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output image or directory")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for super-resolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device)
    debug_print(f"Using device: {device}")
    
    model = load_model(args.model, args.model_type, args.scale, device)
    
    # Process input
    if os.path.isdir(args.input):
        debug_print(f"Processing directory: {args.input}")
        process_directory(model, args.input, args.output, args.model_type, args.scale, device)
    else:
        debug_print(f"Processing single image: {args.input}")
        # Preprocess image
        lr_img = preprocess_image(args.input)
        
        # Apply super-resolution
        sr_img = super_resolve_image(model, lr_img, args.model_type, args.scale, device)
        
        # Save output image
        debug_print(f"Saving super-resolved image to: {args.output}")
        save_image(sr_img[0], args.output)
        
        debug_print("Processing complete")

if __name__ == "__main__":
    main() 