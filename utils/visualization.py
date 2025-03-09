import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import cv2
from PIL import Image

# Debug flag
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a numpy array for visualization
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        numpy.ndarray: Numpy array of shape (H, W, C) or (B, H, W, C)
    """
    debug_print(f"Converting tensor to numpy, input shape: {tensor.shape}")
    
    # Make sure tensor is on CPU and detached from computation graph
    tensor = tensor.detach().cpu()
    
    # Handle single images vs. batches
    if len(tensor.shape) == 4:  # (B, C, H, W)
        debug_print("Transposing batch of images")
        # Convert from (B, C, H, W) to (B, H, W, C)
        img = tensor.permute(0, 2, 3, 1).numpy()
    else:  # (C, H, W)
        debug_print("Transposing single image")
        # Convert from (C, H, W) to (H, W, C)
        img = tensor.permute(1, 2, 0).numpy()
    
    # Clip values to [0, 1] range
    img = np.clip(img, 0, 1)
    debug_print(f"Tensor converted to numpy array with shape: {img.shape}")
    
    return img


def save_image(img, path, normalize=True):
    """
    Save an image to disk
    
    Args:
        img: Image to save, can be a PyTorch tensor or numpy array
        path (str): Path to save the image
        normalize (bool): Whether to normalize the image to [0, 255]
    """
    debug_print(f"Saving image to {path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Convert tensor to numpy if needed
    if isinstance(img, torch.Tensor):
        debug_print("Converting tensor to numpy for saving")
        img = tensor_to_numpy(img)
    
    # Normalize to [0, 255] if needed
    if normalize:
        debug_print("Normalizing image to [0, 255]")
        img = (img * 255).astype(np.uint8)
    
    # Handle different image formats
    if path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
        debug_print("Saving as PNG/JPG using OpenCV")
        # OpenCV expects BGR, so convert from RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            debug_print("Converting RGB to BGR for OpenCV")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
    elif path.endswith('.tif') or path.endswith('.tiff'):
        debug_print("Saving as TIFF using PIL")
        # PIL expects RGB
        pil_img = Image.fromarray(img)
        pil_img.save(path)
    else:
        debug_print("Unknown file format, saving as PNG using OpenCV")
        # Default to PNG with OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            debug_print("Converting RGB to BGR for OpenCV")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + '.png', img)
    
    debug_print(f"Image saved successfully to {path}")


def plot_comparison(lr_img, sr_img, hr_img=None, figsize=(15, 5), titles=None):
    """
    Plot low-resolution, super-resolved, and high-resolution images side by side
    
    Args:
        lr_img (torch.Tensor or numpy.ndarray): Low-resolution image
        sr_img (torch.Tensor or numpy.ndarray): Super-resolved image
        hr_img (torch.Tensor or numpy.ndarray, optional): High-resolution image
        figsize (tuple): Figure size
        titles (list): List of titles for the subplots
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    debug_print("Creating comparison plot")
    
    # Convert tensors to numpy if needed
    if isinstance(lr_img, torch.Tensor):
        debug_print("Converting LR tensor to numpy")
        lr_img = tensor_to_numpy(lr_img)
    
    if isinstance(sr_img, torch.Tensor):
        debug_print("Converting SR tensor to numpy")
        sr_img = tensor_to_numpy(sr_img)
    
    if hr_img is not None and isinstance(hr_img, torch.Tensor):
        debug_print("Converting HR tensor to numpy")
        hr_img = tensor_to_numpy(hr_img)
    
    # Create figure
    n_plots = 3 if hr_img is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Set default titles if not provided
    if titles is None:
        titles = ['Low Resolution', 'Super Resolved', 'High Resolution']
    
    # Plot images
    axes[0].imshow(lr_img)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(sr_img)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    if hr_img is not None:
        axes[2].imshow(hr_img)
        axes[2].set_title(titles[2])
        axes[2].axis('off')
    
    plt.tight_layout()
    debug_print("Comparison plot created")
    
    return fig


def plot_training_progress(train_losses, val_metrics, figsize=(15, 5)):
    """
    Plot training progress
    
    Args:
        train_losses (list): List of training losses
        val_metrics (dict): Dictionary with validation metrics
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    debug_print("Creating training progress plot")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot training loss
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    
    # Plot validation loss
    if 'loss' in val_metrics:
        axes[1].plot(val_metrics['loss'])
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
    
    # Plot PSNR
    if 'psnr' in val_metrics:
        axes[2].plot(val_metrics['psnr'])
        axes[2].set_title('PSNR')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('PSNR (dB)')
    
    plt.tight_layout()
    debug_print("Training progress plot created")
    
    return fig


def create_grid(images, nrow=8, padding=2, normalize=True):
    """
    Create a grid of images
    
    Args:
        images (torch.Tensor): Batch of images of shape (B, C, H, W)
        nrow (int): Number of images per row
        padding (int): Padding between images
        normalize (bool): Whether to normalize images to [0, 1]
        
    Returns:
        torch.Tensor: Grid of images
    """
    debug_print(f"Creating grid of images, input shape: {images.shape}")
    
    # Get dimensions
    b, c, h, w = images.shape
    
    # Compute grid dimensions
    ncol = (b + nrow - 1) // nrow
    
    # Create empty grid
    grid = torch.zeros(c, h * ncol + padding * (ncol - 1), w * nrow + padding * (nrow - 1))
    
    # Fill grid with images
    for i in range(b):
        row = i // nrow
        col = i % nrow
        grid[:, row * (h + padding):(row * (h + padding) + h), col * (w + padding):(col * (w + padding) + w)] = images[i]
    
    debug_print(f"Grid created with shape: {grid.shape}")
    
    return grid


def visualize_attention_maps(model, image, layer_name=None):
    """
    Visualize attention maps for a given model and image
    
    Args:
        model: Model with attention layers
        image (torch.Tensor): Input image
        layer_name (str, optional): Name of the layer to visualize
        
    Returns:
        matplotlib.figure.Figure: Figure with attention maps
    """
    debug_print("Attention map visualization not implemented yet")
    # This is a placeholder for future implementation
    pass 