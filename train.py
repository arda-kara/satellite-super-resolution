import os
# Disable TensorFlow oneDNN optimizations to avoid warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import numpy as np
import random
from datetime import datetime

from models.esrgan import RRDBNet, Discriminator, VGGFeatureExtractor
from models.srcnn import SRCNN, FSRCNN
from utils.data_preprocessing import get_data_loaders
from utils.metrics import calculate_psnr, calculate_ssim, PerceptualLoss, GANLoss, evaluate_model
from utils.visualization import plot_comparison, save_image, plot_training_progress, tensor_to_numpy
from utils.losses import AdversarialLoss

# Debug flag
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_esrgan(config):
    """
    Train ESRGAN model
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        None
    """
    # Set random seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        config['data']['lr_dir'],
        config['data']['hr_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create models
    netG = RRDBNet(
        in_channels=3,
        out_channels=3,
        nf=config['model']['nf'],
        nb=config['model']['nb'],
        scale=config['model']['scale']
    ).to(device)
    
    netD = Discriminator(in_channels=3, nf=config['model']['nf']).to(device)
    
    # Initialize weights
    if config['model']['pretrained_g']:
        netG.load_state_dict(torch.load(config['model']['pretrained_g']))
        print(f"Loaded pretrained generator from {config['model']['pretrained_g']}")
    
    if config['model']['pretrained_d']:
        netD.load_state_dict(torch.load(config['model']['pretrained_d']))
        print(f"Loaded pretrained discriminator from {config['model']['pretrained_d']}")
    
    # Create optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=config['training']['lr_g'], betas=(0.9, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=config['training']['lr_d'], betas=(0.9, 0.99))
    
    # Create schedulers
    schedulerG = optim.lr_scheduler.MultiStepLR(
        optimizerG, 
        milestones=config['training']['lr_steps'], 
        gamma=config['training']['lr_gamma']
    )
    
    schedulerD = optim.lr_scheduler.MultiStepLR(
        optimizerD, 
        milestones=config['training']['lr_steps'], 
        gamma=config['training']['lr_gamma']
    )
    
    # Create losses
    pixel_criterion = nn.L1Loss().to(device)
    perceptual_criterion = PerceptualLoss(feature_layer=35).to(device)
    gan_criterion = GANLoss(gan_type='vanilla').to(device)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # Create directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['sample_dir'], exist_ok=True)
    
    # Training loop
    total_iters = 0
    train_losses = {'g_loss': [], 'd_loss': []}
    val_metrics = {'psnr': [], 'ssim': []}
    
    for epoch in range(config['training']['num_epochs']):
        # Training
        netG.train()
        netD.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        for batch in pbar:
            total_iters += 1
            
            # Get data
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Train Discriminator
            optimizerD.zero_grad()
            
            # Generate fake image
            with torch.no_grad():
                fake = netG(lr)
                
            # Real
            real_pred = netD(hr)
            l_d_real = gan_criterion(real_pred, True)
            
            # Fake
            fake_pred = netD(fake.detach())
            l_d_fake = gan_criterion(fake_pred, False)
            
            # Combined loss
            l_d_total = l_d_real + l_d_fake
            
            l_d_total.backward()
            optimizerD.step()
            
            # Train Generator
            optimizerG.zero_grad()
            
            # Pixel loss
            l_g_pix = pixel_criterion(fake, hr)
            
            # Perceptual loss
            l_g_percep = perceptual_criterion(fake, hr)
            
            # GAN loss
            fake_pred = netD(fake)
            l_g_gan = gan_criterion(fake_pred, True)
            
            # Combined loss
            l_g_total = (
                config['training']['pixel_weight'] * l_g_pix + 
                config['training']['perceptual_weight'] * l_g_percep + 
                config['training']['gan_weight'] * l_g_gan
            )
            
            l_g_total.backward()
            optimizerG.step()
            
            # Update progress bar
            pbar.set_postfix({
                'g_loss': l_g_total.item(),
                'd_loss': l_d_total.item()
            })
            
            # Log losses
            train_losses['g_loss'].append(l_g_total.item())
            train_losses['d_loss'].append(l_d_total.item())
            
            # Log to tensorboard
            writer.add_scalar('train/g_loss', l_g_total.item(), total_iters)
            writer.add_scalar('train/d_loss', l_d_total.item(), total_iters)
            writer.add_scalar('train/g_pix_loss', l_g_pix.item(), total_iters)
            writer.add_scalar('train/g_percep_loss', l_g_percep.item(), total_iters)
            writer.add_scalar('train/g_gan_loss', l_g_gan.item(), total_iters)
            
            # Generate samples
            if total_iters % config['training']['sample_interval'] == 0:
                with torch.no_grad():
                    # Get a sample batch
                    sample_lr = next(iter(val_loader))['lr'].to(device)
                    sample_hr = next(iter(val_loader))['hr'].to(device)
                    sample_sr = netG(sample_lr)
                    
                    # Save images
                    for i in range(min(5, sample_lr.size(0))):
                        save_image(
                            sample_lr[i],
                            os.path.join(config['training']['sample_dir'], f'iter_{total_iters}_lr_{i}.png')
                        )
                        save_image(
                            sample_sr[i],
                            os.path.join(config['training']['sample_dir'], f'iter_{total_iters}_sr_{i}.png')
                        )
                        save_image(
                            sample_hr[i],
                            os.path.join(config['training']['sample_dir'], f'iter_{total_iters}_hr_{i}.png')
                        )
                        
                    # Log images to tensorboard
                    writer.add_images('val/lr', sample_lr, total_iters)
                    writer.add_images('val/sr', sample_sr, total_iters)
                    writer.add_images('val/hr', sample_hr, total_iters)
        
        # Update learning rate
        schedulerG.step()
        schedulerD.step()
        
        # Validation
        metrics = evaluate_model(netG, val_loader, device)
        val_metrics['psnr'].append(metrics['psnr'])
        val_metrics['ssim'].append(metrics['ssim'])
        
        # Log metrics
        writer.add_scalar('val/psnr', metrics['psnr'], epoch)
        writer.add_scalar('val/ssim', metrics['ssim'], epoch)
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save(
                netG.state_dict(),
                os.path.join(config['training']['checkpoint_dir'], f'netG_epoch_{epoch+1}.pth')
            )
            torch.save(
                netD.state_dict(),
                os.path.join(config['training']['checkpoint_dir'], f'netD_epoch_{epoch+1}.pth')
            )
            
            # Plot and save training progress
            fig = plot_training_progress(train_losses, val_metrics)
            fig.savefig(os.path.join(config['training']['log_dir'], f'training_progress_epoch_{epoch+1}.png'))
            plt.close(fig)
    
    # Save final model
    torch.save(
        netG.state_dict(),
        os.path.join(config['training']['checkpoint_dir'], 'netG_final.pth')
    )
    torch.save(
        netD.state_dict(),
        os.path.join(config['training']['checkpoint_dir'], 'netD_final.pth')
    )
    
    # Close tensorboard writer
    writer.close()
    
    print("Training completed!")


def train_srcnn(config):
    """
    Train SRCNN model
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        None
    """
    # Set random seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        config['data']['lr_dir'],
        config['data']['hr_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    if config['model']['type'] == 'srcnn':
        model = SRCNN(num_channels=3).to(device)
    elif config['model']['type'] == 'fsrcnn':
        model = FSRCNN(
            scale_factor=config['model']['scale'],
            num_channels=3
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    # Initialize weights
    if config['model']['pretrained']:
        model.load_state_dict(torch.load(config['model']['pretrained']))
        print(f"Loaded pretrained model from {config['model']['pretrained']}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Create scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config['training']['lr_steps'], 
        gamma=config['training']['lr_gamma']
    )
    
    # Create loss
    criterion = nn.MSELoss().to(device)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # Create directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['sample_dir'], exist_ok=True)
    
    # Training loop
    total_iters = 0
    train_losses = {'loss': []}
    val_metrics = {'psnr': [], 'ssim': []}
    
    for epoch in range(config['training']['num_epochs']):
        # Training
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        for batch in pbar:
            total_iters += 1
            
            # Get data
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Forward pass
            if config['model']['type'] == 'srcnn':
                # SRCNN expects the input to be already upscaled
                lr_upscaled = nn.functional.interpolate(
                    lr, 
                    scale_factor=config['model']['scale'], 
                    mode='bicubic', 
                    align_corners=False
                )
                sr = model(lr_upscaled)
            else:
                sr = model(lr)
            
            # Calculate loss
            loss = criterion(sr, hr)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log loss
            train_losses['loss'].append(loss.item())
            
            # Log to tensorboard
            writer.add_scalar('train/loss', loss.item(), total_iters)
            
            # Generate samples
            if total_iters % config['training']['sample_interval'] == 0:
                with torch.no_grad():
                    # Get a sample batch
                    sample_lr = next(iter(val_loader))['lr'].to(device)
                    sample_hr = next(iter(val_loader))['hr'].to(device)
                    
                    if config['model']['type'] == 'srcnn':
                        sample_lr_upscaled = nn.functional.interpolate(
                            sample_lr, 
                            scale_factor=config['model']['scale'], 
                            mode='bicubic', 
                            align_corners=False
                        )
                        sample_sr = model(sample_lr_upscaled)
                    else:
                        sample_sr = model(sample_lr)
                    
                    # Save images
                    for i in range(min(5, sample_lr.size(0))):
                        save_image(
                            sample_lr[i],
                            os.path.join(config['training']['sample_dir'], f'iter_{total_iters}_lr_{i}.png')
                        )
                        save_image(
                            sample_sr[i],
                            os.path.join(config['training']['sample_dir'], f'iter_{total_iters}_sr_{i}.png')
                        )
                        save_image(
                            sample_hr[i],
                            os.path.join(config['training']['sample_dir'], f'iter_{total_iters}_hr_{i}.png')
                        )
                        
                    # Log images to tensorboard
                    writer.add_images('val/lr', sample_lr, total_iters)
                    writer.add_images('val/sr', sample_sr, total_iters)
                    writer.add_images('val/hr', sample_hr, total_iters)
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        psnr_values = []
        ssim_values = []
        
        with torch.no_grad():
            for batch in val_loader:
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                
                if config['model']['type'] == 'srcnn':
                    lr_upscaled = nn.functional.interpolate(
                        lr, 
                        scale_factor=config['model']['scale'], 
                        mode='bicubic', 
                        align_corners=False
                    )
                    sr = model(lr_upscaled)
                else:
                    sr = model(lr)
                
                # Calculate loss
                loss = criterion(sr, hr)
                val_loss += loss.item()
                
                # Calculate metrics
                psnr = calculate_psnr(sr, hr)
                ssim = calculate_ssim(sr, hr)
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
        
        # Average metrics
        val_loss /= len(val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        val_metrics['psnr'].append(avg_psnr)
        val_metrics['ssim'].append(avg_ssim)
        
        # Log metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/psnr', avg_psnr, epoch)
        writer.add_scalar('val/ssim', avg_ssim, epoch)
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Loss: {val_loss:.6f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(config['training']['checkpoint_dir'], f'model_epoch_{epoch+1}.pth')
            )
            
            # Plot and save training progress
            fig = plot_training_progress({'loss': train_losses['loss']}, val_metrics)
            fig.savefig(os.path.join(config['training']['log_dir'], f'training_progress_epoch_{epoch+1}.png'))
            plt.close(fig)
    
    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(config['training']['checkpoint_dir'], 'model_final.pth')
    )
    
    # Close tensorboard writer
    writer.close()
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train super-resolution models")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory with low-resolution images")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory with high-resolution images")
    parser.add_argument("--model_type", type=str, default="esrgan", choices=["esrgan", "srcnn", "fsrcnn"], 
                        help="Model architecture to use")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--sample_dir", type=str, default="samples", help="Directory to save sample images")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type)
    log_dir = os.path.join(args.log_dir, f"{args.model_type}_{timestamp}")
    sample_dir = os.path.join(args.sample_dir, args.model_type)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(args.lr_dir, args.hr_dir, args.batch_size)
    debug_print(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Create model
    device = torch.device(args.device)
    debug_print(f"Using device: {device}")
    
    if args.model_type == "esrgan":
        model = RRDBNet(3, 3, 64, 23, gc=32)
        debug_print("Created ESRGAN model")
    elif args.model_type == "srcnn":
        model = SRCNN()
        debug_print("Created SRCNN model")
    elif args.model_type == "fsrcnn":
        model = FSRCNN(scale_factor=args.scale)
        debug_print("Created FSRCNN model")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create loss function
    if args.model_type == "esrgan":
        criterion = PerceptualLoss(device)
        debug_print("Using Perceptual Loss")
    else:
        criterion = nn.MSELoss()
        debug_print("Using MSE Loss")
    
    # Train model
    train(model, train_loader, val_loader, optimizer, criterion, device, args.num_epochs, 
          checkpoint_dir, log_dir, sample_dir, args.model_type)

if __name__ == "__main__":
    main() 