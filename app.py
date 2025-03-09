import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import time

from models.esrgan import RRDBNet
from models.srcnn import SRCNN, FSRCNN
from utils.visualization import save_image


# Create FastAPI app
app = FastAPI(title="Satellite Image Super-Resolution API")

# Create directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/esrgan/netG_final.pth")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "esrgan")
SCALE_FACTOR = int(os.environ.get("SCALE_FACTOR", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "static/results"

# Debug flag
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

# Load model
debug_print(f"Loading model: {MODEL_TYPE} from {MODEL_PATH}")
if MODEL_TYPE == "esrgan":
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        nf=64,
        nb=23,
        scale=SCALE_FACTOR
    ).to(DEVICE)
elif MODEL_TYPE == "srcnn":
    model = SRCNN(num_channels=3).to(DEVICE)
elif MODEL_TYPE == "fsrcnn":
    model = FSRCNN(scale_factor=SCALE_FACTOR, num_channels=3).to(DEVICE)
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Load weights if model file exists
if os.path.exists(MODEL_PATH):
    if DEVICE == "cuda" and torch.cuda.is_available():
        state_dict = torch.load(MODEL_PATH)
    else:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Warning: Model file {MODEL_PATH} not found. Using untrained model.")

# Set model to evaluation mode
model.eval()


def preprocess_image(image, scale_factor=None):
    """
    Preprocess image for inference
    
    Args:
        image (PIL.Image): Input image
        scale_factor (int, optional): Scale factor for SRCNN
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    debug_print(f"Preprocessing image: {image.size}, mode: {image.mode}")
    
    # Convert PIL image to numpy array
    img = np.array(image)
    debug_print(f"Image shape after conversion to numpy: {img.shape}")
    
    # Handle RGBA images by converting to RGB
    if len(img.shape) == 3 and img.shape[2] == 4:
        debug_print("Converting RGBA to RGB")
        # Convert RGBA to RGB by removing the alpha channel
        img = img[:, :, :3]
    
    # Ensure we're working with RGB (not BGR)
    # PIL already uses RGB, so no conversion needed if coming from PIL
    # Only convert if the image came from OpenCV (which uses BGR)
    if 'cv2' in str(type(image)):
        debug_print("Converting BGR to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if scale_factor is not None:
        h, w = img.shape[:2]
        debug_print(f"Resizing image from {w}x{h} to {w//scale_factor}x{h//scale_factor}")
        img = cv2.resize(img, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    debug_print(f"Tensor shape after preprocessing: {img.shape}")
    
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
    debug_print(f"Starting super-resolution with model: {model_type}")
    start_time = time.time()
    
    with torch.no_grad():
        lr_img = lr_img.to(device)
        debug_print(f"Input tensor shape: {lr_img.shape}, device: {device}")
        
        if model_type == 'srcnn':
            # SRCNN expects the input to be already upscaled
            debug_print("Upscaling input for SRCNN")
            lr_upscaled = nn.functional.interpolate(
                lr_img, 
                scale_factor=scale_factor, 
                mode='bicubic', 
                align_corners=False
            )
            sr_img = model(lr_upscaled)
        else:
            sr_img = model(lr_img)
    
    elapsed_time = time.time() - start_time
    debug_print(f"Super-resolution completed in {elapsed_time:.2f} seconds")
    debug_print(f"Output tensor shape: {sr_img.shape}")
    
    return sr_img


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL image
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        PIL.Image: PIL image
    """
    debug_print(f"Converting tensor to PIL image, tensor shape: {tensor.shape}")
    
    # Convert to numpy
    img = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Convert to PIL image
    img = Image.fromarray(img)
    debug_print(f"PIL image size: {img.size}, mode: {img.mode}")
    
    return img


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the index page
    """
    debug_print("Rendering index page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    """
    Upscale an image
    
    Args:
        file (UploadFile): Input image file
        
    Returns:
        dict: Dictionary with original and upscaled image paths
    """
    debug_print(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        debug_print(f"Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read image
        debug_print("Reading image file")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save original image
        original_path = f"static/uploads/{file.filename}"
        debug_print(f"Saving original image to {original_path}")
        image.save(original_path)
        
        # Preprocess image
        debug_print("Preprocessing image")
        lr_img = preprocess_image(image)
        
        # Super-resolve image
        debug_print("Super-resolving image")
        sr_img = super_resolve_image(model, lr_img, MODEL_TYPE, SCALE_FACTOR, DEVICE)
        
        # Convert to PIL image
        debug_print("Converting super-resolved image to PIL format")
        sr_pil = tensor_to_pil(sr_img)
        
        # Save super-resolved image
        sr_path = f"static/results/sr_{file.filename}"
        debug_print(f"Saving super-resolved image to {sr_path}")
        sr_pil.save(sr_path)
        
        # Convert images to base64 for display
        debug_print("Converting images to base64")
        with open(original_path, "rb") as f:
            original_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        with open(sr_path, "rb") as f:
            sr_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        debug_print("Returning response")
        return {
            "original_path": original_path,
            "sr_path": sr_path,
            "original_base64": original_base64,
            "sr_base64": sr_base64
        }
    except Exception as e:
        debug_print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/info")
async def info():
    """
    Get model information
    
    Returns:
        dict: Dictionary with model information
    """
    debug_print("Returning model information")
    return {
        "model_type": MODEL_TYPE,
        "scale_factor": SCALE_FACTOR,
        "device": str(DEVICE),
        "model_path": MODEL_PATH,
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 