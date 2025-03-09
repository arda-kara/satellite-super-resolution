import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import json

from models.esrgan import RRDBNet
from models.srcnn import SRCNN, FSRCNN
from utils.visualization import tensor_to_numpy

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

def process_image(input_path, output_path, model_type='esrgan', scale_factor=4, device='cuda'):
    """
    Process an image for Unity integration
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to output image
        model_type (str): Model type ('esrgan', 'srcnn', or 'fsrcnn')
        scale_factor (int): Scale factor
        device (str): Device to use
        
    Returns:
        dict: Dictionary with processing information
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    # Load model
    if model_type == 'esrgan':
        model = RRDBNet(
            in_channels=3,
            out_channels=3,
            nf=64,
            nb=23,
            scale=scale_factor
        ).to(device)
        model_path = 'checkpoints/esrgan/netG_final.pth'
    elif model_type == 'srcnn':
        model = SRCNN(num_channels=3).to(device)
        model_path = 'checkpoints/srcnn/model_final.pth'
    elif model_type == 'fsrcnn':
        model = FSRCNN(scale_factor=scale_factor, num_channels=3).to(device)
        model_path = 'checkpoints/fsrcnn/model_final.pth'
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
    
    # Preprocess image
    lr_img = preprocess_image(input_path)
    
    # Super-resolve image
    sr_img = super_resolve_image(model, lr_img, model_type, scale_factor, device)
    
    # Convert to numpy
    sr_np = tensor_to_numpy(sr_img)
    
    # Convert to uint8
    sr_np = (sr_np * 255).astype(np.uint8)
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))
    
    # Return processing information
    return {
        'input_path': input_path,
        'output_path': output_path,
        'model_type': model_type,
        'scale_factor': scale_factor,
        'device': str(device),
        'input_shape': list(lr_img.shape),
        'output_shape': list(sr_img.shape)
    }

def process_directory(input_dir, output_dir, model_type='esrgan', scale_factor=4, device='cuda'):
    """
    Process all images in a directory for Unity integration
    
    Args:
        input_dir (str): Path to input directory
        output_dir (str): Path to output directory
        model_type (str): Model type ('esrgan', 'srcnn', or 'fsrcnn')
        scale_factor (int): Scale factor
        device (str): Device to use
        
    Returns:
        list: List of dictionaries with processing information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image_files.extend(
            [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(ext)]
        )
    
    # Process each image
    results = []
    for input_path in image_files:
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        result = process_image(input_path, output_path, model_type, scale_factor, device)
        results.append(result)
        print(f"Processed {input_path} -> {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Super-resolution for Unity integration")
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output image or directory')
    parser.add_argument('--model', type=str, default='esrgan', choices=['esrgan', 'srcnn', 'fsrcnn'], help='Model type')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--json', type=str, help='Path to save processing information as JSON')
    
    args = parser.parse_args()
    
    # Process image or directory
    if os.path.isdir(args.input):
        results = process_directory(args.input, args.output, args.model, args.scale, args.device)
    else:
        results = [process_image(args.input, args.output, args.model, args.scale, args.device)]
    
    # Save processing information as JSON
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
    
    print("Processing completed!")

if __name__ == "__main__":
    main() 