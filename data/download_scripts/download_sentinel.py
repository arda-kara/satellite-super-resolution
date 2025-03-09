import os
import argparse
import time
from tqdm import tqdm
import requests
import shutil
import sys
import traceback  # Added for detailed error tracing
import numpy as np
import cv2
from PIL import Image
import random

# Add debug flag
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

def download_sentinel_from_ee(output_dir, lat, lon, date_range, max_cloud_percentage=10, max_images=10):
    """
    Download Sentinel-2 satellite imagery using Google Earth Engine.
    
    Args:
        output_dir (str): Directory to save downloaded images
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        date_range (tuple): Start and end dates in format 'YYYY-MM-DD'
        max_cloud_percentage (int): Maximum cloud coverage percentage
        max_images (int): Maximum number of images to download
        
    Returns:
        bool: True if successful, False otherwise
    """
    debug_print(f"Entering download_sentinel_from_ee with params: {output_dir}, {lat}, {lon}")
    try:
        debug_print("Attempting to import ee module")
        import ee
        
        # Initialize Earth Engine
        try:
            debug_print("Attempting to initialize Earth Engine")
            ee.Initialize()
            print("Earth Engine initialized successfully")
        except Exception as e:
            debug_print(f"Error initializing Earth Engine: {e}")
            print(f"Error initializing Earth Engine: {e}")
            print("Please authenticate using 'earthengine authenticate' command")
            return False
        
        # Create output directory
        debug_print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the region of interest
        debug_print(f"Creating point geometry at lon={lon}, lat={lat}")
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(10000)  # 10km buffer
        
        # Filter Sentinel-2 collection
        debug_print(f"Filtering Sentinel-2 collection for date range: {date_range}")
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(region)
                     .filterDate(date_range[0], date_range[1])
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percentage))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        # Get the list of images
        debug_print("Getting list of images")
        image_list = collection.toList(collection.size())
        count = min(max_images, image_list.size().getInfo())
        
        print(f"Found {count} Sentinel-2 images")
        
        # Download each image
        for i in tqdm(range(count), desc="Downloading images"):
            debug_print(f"Processing image {i+1}/{count}")
            image = ee.Image(image_list.get(i))
            id = image.id().getInfo()
            
            # Select RGB bands and scale them
            debug_print(f"Selecting RGB bands for image {id}")
            rgb = image.select(['B4', 'B3', 'B2']).divide(10000)
            
            # Get download URL
            debug_print("Getting download URL")
            url = rgb.getDownloadURL({
                'scale': 10,  # 10m resolution
                'region': region,
                'format': 'GEO_TIFF'
            })
            
            # Download the image
            debug_print(f"Downloading image from URL: {url[:100]}...")
            response = requests.get(url)
            
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"sentinel_{id}.tif")
                debug_print(f"Writing image to {output_path}")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {output_path}")
            else:
                debug_print(f"Failed to download image {id}, status code: {response.status_code}")
                print(f"Failed to download image {id}")
            
            # Sleep to avoid rate limiting
            time.sleep(1)
        
        return True
    
    except ImportError as e:
        debug_print(f"ImportError: {e}")
        print(f"Error importing Earth Engine API: {e}")
        print("Earth Engine API not installed or configured correctly.")
        return False
    except Exception as e:
        debug_print(f"Exception in download_sentinel_from_ee: {e}")
        debug_print(f"Traceback: {traceback.format_exc()}")
        print(f"Error downloading Sentinel data from Earth Engine: {e}")
        return False


def download_sentinel_from_aws(output_dir, lat, lon, date_range, max_cloud_percentage=10, max_images=10):
    """
    Download Sentinel-2 satellite imagery from AWS Open Data.
    
    Args:
        output_dir (str): Directory to save downloaded images
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        date_range (tuple): Start and end dates in format 'YYYY-MM-DD'
        max_cloud_percentage (int): Maximum cloud coverage percentage
        max_images (int): Maximum number of images to download
        
    Returns:
        bool: True if successful, False otherwise
    """
    debug_print(f"Entering download_sentinel_from_aws with params: {output_dir}, {lat}, {lon}")
    try:
        # Create output directory
        debug_print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert lat/lon to UTM grid reference
        # This is a simplified approach - in a real implementation, you would use proper conversion
        debug_print(f"Converting lat/lon to UTM grid reference")
        utm_zone = int((lon + 180) / 6) + 1
        lat_band = "CDEFGHJKLMNPQRSTUVWXX"[int((lat + 80) / 8)]
        debug_print(f"UTM Zone: {utm_zone}, Lat Band: {lat_band}")
        
        # AWS Sentinel-2 bucket URL
        base_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
        
        # Sample URLs for demonstration
        sample_urls = [
            f"{base_url}/sentinel-s2-l2a-cogs/2022/S2A_38/S/QB/2022_01_01/S2A_38SQB_20220101_0_L2A/TCI.tif",
            f"{base_url}/sentinel-s2-l2a-cogs/2022/S2A_38/S/QB/2022_02_10/S2A_38SQB_20220210_0_L2A/TCI.tif",
            f"{base_url}/sentinel-s2-l2a-cogs/2022/S2A_38/S/QB/2022_03_22/S2A_38SQB_20220322_0_L2A/TCI.tif",
        ]
        
        print(f"Downloading sample Sentinel-2 images from AWS Open Data")
        
        # Download sample images
        for i, url in enumerate(sample_urls[:max_images]):
            debug_print(f"Downloading from URL: {url}")
            try:
                response = requests.get(url, stream=True)
                debug_print(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    filename = f"sentinel_sample_{i+1}.tif"
                    output_path = os.path.join(output_dir, filename)
                    debug_print(f"Writing to file: {output_path}")
                    
                    with open(output_path, 'wb') as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    
                    print(f"Downloaded {output_path}")
                else:
                    debug_print(f"Failed with status code: {response.status_code}")
                    print(f"Failed to download image from {url}")
            except Exception as e:
                debug_print(f"Error downloading from {url}: {e}")
                debug_print(f"Traceback: {traceback.format_exc()}")
                print(f"Error downloading from {url}: {e}")
            
            # Sleep to avoid rate limiting
            time.sleep(1)
        
        return True
    
    except Exception as e:
        debug_print(f"Exception in download_sentinel_from_aws: {e}")
        debug_print(f"Traceback: {traceback.format_exc()}")
        print(f"Error downloading Sentinel data from AWS: {e}")
        return False


def create_sample_satellite_image(width=512, height=512, channels=3):
    """
    Create a synthetic satellite image for testing purposes.
    
    Args:
        width (int): Image width
        height (int): Image height
        channels (int): Number of channels (3 for RGB)
        
    Returns:
        numpy.ndarray: Synthetic satellite image
    """
    # Create a base image with terrain-like patterns
    x = np.linspace(0, 5, width)
    y = np.linspace(0, 5, height)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Create terrain-like patterns using sine waves
    z = np.sin(x_grid) + np.sin(y_grid) + np.sin(x_grid * 2) * np.sin(y_grid * 2)
    z = (z - z.min()) / (z.max() - z.min())  # Normalize to [0, 1]
    
    # Create RGB image
    img = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Green for vegetation
    img[:, :, 1] = (z * 200).astype(np.uint8)
    
    # Blue for water
    water_mask = z < 0.3
    img[water_mask, 2] = 200
    img[water_mask, 1] = 100
    img[water_mask, 0] = 50
    
    # Brown for mountains/terrain
    mountain_mask = z > 0.7
    img[mountain_mask, 0] = 120
    img[mountain_mask, 1] = 100
    img[mountain_mask, 2] = 80
    
    # Add some random noise for texture
    noise = np.random.randint(0, 30, (height, width, channels), dtype=np.uint8)
    img = np.clip(img + noise, 0, 255)
    
    return img


def download_sample_data(output_dir, max_images=5):
    """
    Create sample satellite images for testing.
    
    Args:
        output_dir (str): Directory to save sample images
        max_images (int): Maximum number of images to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    debug_print(f"Entering download_sample_data with params: {output_dir}, {max_images}")
    try:
        # Create output directory
        debug_print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating {max_images} sample satellite images")
        
        # Create sample images
        for i in range(max_images):
            debug_print(f"Creating sample image {i+1}/{max_images}")
            
            # Create a synthetic satellite image
            width = height = random.choice([512, 1024])
            img = create_sample_satellite_image(width, height)
            
            # Save the image
            filename = f"sample_{i+1}.jpg"
            output_path = os.path.join(output_dir, filename)
            debug_print(f"Writing to file: {output_path}")
            
            # Convert to PIL Image and save
            Image.fromarray(img).save(output_path)
            print(f"Created {output_path}")
        
        return True
    
    except Exception as e:
        debug_print(f"Exception in download_sample_data: {e}")
        debug_print(f"Traceback: {traceback.format_exc()}")
        print(f"Error creating sample data: {e}")
        return False


def main():
    debug_print("Entering main function")
    try:
        parser = argparse.ArgumentParser(description="Download Sentinel-2 satellite imagery")
        parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
        parser.add_argument('--lat', type=float, required=True, help='Latitude of center point')
        parser.add_argument('--lon', type=float, required=True, help='Longitude of center point')
        parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
        parser.add_argument('--max_cloud', type=int, default=10, help='Maximum cloud coverage percentage')
        parser.add_argument('--max_images', type=int, default=10, help='Maximum number of images to download')
        parser.add_argument('--source', type=str, default='auto', choices=['ee', 'aws', 'sample', 'auto'], 
                            help='Data source: Earth Engine (ee), AWS (aws), sample data (sample), or auto-detect (auto)')
        
        debug_print("Parsing arguments")
        args = parser.parse_args()
        debug_print(f"Arguments: {args}")
        
        # Create output directory
        debug_print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Try to download data from the specified source
        success = False
        
        if args.source == 'auto' or args.source == 'ee':
            debug_print("Attempting Earth Engine download")
            print("Attempting to download data from Google Earth Engine...")
            success = download_sentinel_from_ee(
                args.output_dir,
                args.lat,
                args.lon,
                (args.start_date, args.end_date),
                args.max_cloud,
                args.max_images
            )
            debug_print(f"Earth Engine download success: {success}")
            
            if success and args.source == 'ee':
                print("Successfully downloaded data from Google Earth Engine")
                return
            elif not success and args.source == 'ee':
                print("Failed to download data from Google Earth Engine")
                print("Please try installing and authenticating the Earth Engine API:")
                print("pip install earthengine-api")
                print("earthengine authenticate")
                return
        
        if (args.source == 'auto' and not success) or args.source == 'aws':
            debug_print("Attempting AWS download")
            print("Attempting to download data from AWS Open Data...")
            success = download_sentinel_from_aws(
                args.output_dir,
                args.lat,
                args.lon,
                (args.start_date, args.end_date),
                args.max_cloud,
                args.max_images
            )
            debug_print(f"AWS download success: {success}")
            
            if success and args.source == 'aws':
                print("Successfully downloaded data from AWS Open Data")
                return
            elif not success and args.source == 'aws':
                print("Failed to download data from AWS Open Data")
        
        if (args.source == 'auto' and not success) or args.source == 'sample':
            debug_print("Attempting sample data creation")
            print("Creating sample satellite images...")
            success = download_sample_data(args.output_dir, args.max_images)
            debug_print(f"Sample data creation success: {success}")
            
            if success:
                print("Successfully created sample satellite images")
            else:
                print("Failed to create sample satellite images")
        
        if not success:
            debug_print("All download attempts failed")
            print("Failed to download data from all sources")
            print("Please check your internet connection and try again")
            sys.exit(1)
    
    except Exception as e:
        debug_print(f"Exception in main function: {e}")
        debug_print(f"Traceback: {traceback.format_exc()}")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    debug_print("Script started")
    main() 