import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Debug flag
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

def create_lr_hr_pairs(hr_dir, lr_dir, hr_output_dir, scale_factor=4, patch_size=256, stride=200):
    """
    Create low-resolution and high-resolution image pairs from high-resolution images.
    
    Args:
        hr_dir (str): Directory containing original high-resolution images
        lr_dir (str): Directory to save low-resolution patches
        hr_output_dir (str): Directory to save high-resolution patches
        scale_factor (int): Downsampling factor
        patch_size (int): Size of the high-resolution patches to extract
        stride (int): Stride between patches
        
    Returns:
        None
    """
    debug_print(f"Creating LR-HR pairs from {hr_dir}")
    debug_print(f"Saving LR patches to {lr_dir}")
    debug_print(f"Saving HR patches to {hr_output_dir}")
    
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)
    
    hr_files = sorted(glob(os.path.join(hr_dir, "*.tif")) + glob(os.path.join(hr_dir, "*.jpg")) + 
                      glob(os.path.join(hr_dir, "*.png")))
    
    print(f"Found {len(hr_files)} high-resolution images")
    
    for idx, hr_file in enumerate(tqdm(hr_files, desc="Creating LR-HR pairs")):
        # Read high-resolution image
        try:
            if hr_file.endswith('.tif'):
                with rasterio.open(hr_file) as src:
                    hr_img = src.read()
                    # Convert from (C, H, W) to (H, W, C)
                    hr_img = np.transpose(hr_img, (1, 2, 0))
                    debug_print(f"Read TIFF image with shape: {hr_img.shape}")
            else:
                # OpenCV reads in BGR, convert to RGB for consistency
                hr_img = cv2.imread(hr_file)
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                debug_print(f"Read image with shape: {hr_img.shape}, converted BGR to RGB")
        except Exception as e:
            print(f"Error reading {hr_file}: {e}")
            continue
            
        # Skip images that are too small
        if hr_img.shape[0] < patch_size or hr_img.shape[1] < patch_size:
            print(f"Skipping {hr_file} - too small: {hr_img.shape}")
            continue
            
        # Extract patches
        for i in range(0, hr_img.shape[0] - patch_size + 1, stride):
            for j in range(0, hr_img.shape[1] - patch_size + 1, stride):
                # Extract HR patch
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size]
                
                # Create LR patch by downsampling
                lr_patch = cv2.resize(hr_patch, 
                                     (patch_size // scale_factor, patch_size // scale_factor), 
                                     interpolation=cv2.INTER_CUBIC)
                
                # Save patches
                patch_name = f"{os.path.splitext(os.path.basename(hr_file))[0]}_patch_{i}_{j}"
                
                # Save HR patch to hr_output_dir - convert RGB to BGR for OpenCV
                hr_patch_path = os.path.join(hr_output_dir, f"{patch_name}.png")
                cv2.imwrite(hr_patch_path, cv2.cvtColor(hr_patch, cv2.COLOR_RGB2BGR))
                debug_print(f"Saved HR patch to {hr_patch_path}, converted RGB to BGR for saving")
                
                # Save LR patch to lr_dir - convert RGB to BGR for OpenCV
                lr_patch_path = os.path.join(lr_dir, f"{patch_name}.png")
                cv2.imwrite(lr_patch_path, cv2.cvtColor(lr_patch, cv2.COLOR_RGB2BGR))
                debug_print(f"Saved LR patch to {lr_patch_path}, converted RGB to BGR for saving")
                
    print(f"Created LR-HR pairs in {lr_dir} and {hr_output_dir}")


def download_sentinel_data(output_dir, lat, lon, date_range, max_cloud_percentage=10):
    """
    Download Sentinel-2 satellite imagery for a specific location and time range.
    
    Args:
        output_dir (str): Directory to save downloaded images
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        date_range (tuple): Start and end dates in format 'YYYY-MM-DD'
        max_cloud_percentage (int): Maximum cloud coverage percentage
        
    Returns:
        None
    """
    try:
        import ee
        
        # Initialize Earth Engine
        ee.Initialize()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the region of interest
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(10000)  # 10km buffer
        
        # Filter Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(region)
                     .filterDate(date_range[0], date_range[1])
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percentage)))
        
        # Get the list of images
        image_list = collection.toList(collection.size())
        count = image_list.size().getInfo()
        
        print(f"Found {count} Sentinel-2 images")
        
        # Download each image
        for i in range(count):
            image = ee.Image(image_list.get(i))
            id = image.id().getInfo()
            
            # Select RGB bands and scale them
            rgb = image.select(['B4', 'B3', 'B2']).divide(10000)
            
            # Get download URL
            url = rgb.getDownloadURL({
                'scale': 10,  # 10m resolution
                'region': region,
                'format': 'GEO_TIFF'
            })
            
            # Download the image
            import requests
            response = requests.get(url)
            
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"sentinel_{id}.tif")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {output_path}")
            else:
                print(f"Failed to download image {id}")
                
    except ImportError:
        print("Earth Engine API not installed. Please install with: pip install earthengine-api")
    except Exception as e:
        print(f"Error downloading Sentinel data: {e}")


class SatelliteDataset(Dataset):
    """
    Dataset for satellite image super-resolution
    """
    def __init__(self, lr_dir, hr_dir, transform=None):
        """
        Args:
            lr_dir (str): Directory with low-resolution images
            hr_dir (str): Directory with high-resolution images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.lr_files = sorted(glob(os.path.join(lr_dir, "*.png")) + 
                              glob(os.path.join(lr_dir, "*.jpg")) + 
                              glob(os.path.join(lr_dir, "*.tif")))
        self.hr_files = sorted(glob(os.path.join(hr_dir, "*.png")) + 
                              glob(os.path.join(hr_dir, "*.jpg")) + 
                              glob(os.path.join(hr_dir, "*.tif")))
        
        # Ensure matching pairs
        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(f"Number of LR ({len(self.lr_files)}) and HR ({len(self.hr_files)}) images don't match")
            
        self.transform = transform
        debug_print(f"Created SatelliteDataset with {len(self.lr_files)} image pairs")
        
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        # Load images - OpenCV reads in BGR
        lr_img = cv2.imread(self.lr_files[idx])
        hr_img = cv2.imread(self.hr_files[idx])
        
        # Convert from BGR to RGB for consistency
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        debug_print(f"Loaded image pair {idx}, converted BGR to RGB")
        
        # Normalize to [0, 1]
        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0
        
        # Apply transforms if any
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        else:
            # Convert to PyTorch tensors
            lr_img = torch.from_numpy(lr_img).permute(2, 0, 1)
            hr_img = torch.from_numpy(hr_img).permute(2, 0, 1)
        
        return {'lr': lr_img, 'hr': hr_img}


def get_data_loaders(lr_dir, hr_dir, batch_size=8, num_workers=4, train_test_split=0.8):
    """
    Create data loaders for training and validation
    
    Args:
        lr_dir (str): Directory with low-resolution images
        hr_dir (str): Directory with high-resolution images
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        train_test_split (float): Ratio of training data
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    debug_print(f"Creating data loaders with batch size {batch_size}")
    
    # Create dataset
    dataset = SatelliteDataset(lr_dir, hr_dir)
    
    # Split into train and validation sets
    train_size = int(train_test_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    debug_print(f"Split dataset into {train_size} training and {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Satellite image preprocessing")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory with original high-resolution images")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory to save low-resolution patches")
    parser.add_argument("--hr_output_dir", type=str, default=None, help="Directory to save high-resolution patches")
    parser.add_argument("--scale", type=int, default=4, help="Downsampling factor")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches to extract")
    parser.add_argument("--stride", type=int, default=200, help="Stride between patches")
    
    args = parser.parse_args()
    
    # If hr_output_dir is not specified, use the parent directory of lr_dir + "/hr"
    if args.hr_output_dir is None:
        args.hr_output_dir = os.path.join(os.path.dirname(args.lr_dir), "hr")
    
    create_lr_hr_pairs(args.hr_dir, args.lr_dir, args.hr_output_dir, args.scale, args.patch_size, args.stride) 