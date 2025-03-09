import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1 (torch.Tensor or numpy.ndarray): First image
        img2 (torch.Tensor or numpy.ndarray): Second image
        max_val (float): Maximum value of the images
        
    Returns:
        float: PSNR value
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    # If tensors are in NCHW format, convert to NHWC
    if img1.shape[1] == 3 and len(img1.shape) == 4:
        img1 = np.transpose(img1, (0, 2, 3, 1))
    if img2.shape[1] == 3 and len(img2.shape) == 4:
        img2 = np.transpose(img2, (0, 2, 3, 1))
        
    # Calculate PSNR for each image in the batch
    psnr_values = []
    for i in range(img1.shape[0]):
        psnr_values.append(peak_signal_noise_ratio(img1[i], img2[i], data_range=max_val))
        
    return np.mean(psnr_values)


def calculate_ssim(img1, img2, max_val=1.0):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1 (torch.Tensor or numpy.ndarray): First image
        img2 (torch.Tensor or numpy.ndarray): Second image
        max_val (float): Maximum value of the images
        
    Returns:
        float: SSIM value
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    # If tensors are in NCHW format, convert to NHWC
    if img1.shape[1] == 3 and len(img1.shape) == 4:
        img1 = np.transpose(img1, (0, 2, 3, 1))
    if img2.shape[1] == 3 and len(img2.shape) == 4:
        img2 = np.transpose(img2, (0, 2, 3, 1))
        
    # Calculate SSIM for each image in the batch
    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_values.append(structural_similarity(
            img1[i], img2[i], data_range=max_val, channel_axis=2, multichannel=True
        ))
        
    return np.mean(ssim_values)


class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss using VGG19 features
    """
    def __init__(self, feature_layer=35):
        super(PerceptualLoss, self).__init__()
        from models.esrgan import VGGFeatureExtractor
        self.feature_extractor = VGGFeatureExtractor(feature_layer=feature_layer)
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, sr, hr):
        """
        Calculate perceptual loss
        
        Args:
            sr (torch.Tensor): Super-resolution image
            hr (torch.Tensor): High-resolution image
            
        Returns:
            torch.Tensor: Perceptual loss
        """
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.l1_loss(sr_features, hr_features)


class GANLoss(torch.nn.Module):
    """
    GAN loss for ESRGAN
    """
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        
        if self.gan_type == 'vanilla':
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = torch.nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                return -torch.mean(input) if target else torch.mean(input)
            self.loss = wgan_loss
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented')
            
    def get_target_label(self, input, target_is_real):
        """
        Get target label based on the input size
        
        Args:
            input (torch.Tensor): Input tensor
            target_is_real (bool): Whether the target is real
            
        Returns:
            torch.Tensor: Target tensor
        """
        if target_is_real:
            return torch.ones_like(input).fill_(self.real_label_val)
        else:
            return torch.ones_like(input).fill_(self.fake_label_val)
            
    def forward(self, input, target_is_real):
        """
        Calculate GAN loss
        
        Args:
            input (torch.Tensor): Input tensor
            target_is_real (bool): Whether the target is real
            
        Returns:
            torch.Tensor: GAN loss
        """
        target_label = self.get_target_label(input, target_is_real)
        
        if self.gan_type == 'wgan-gp':
            return self.loss(input, target_is_real)
        else:
            return self.loss(input, target_label)


def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on a dataset
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            sr = model(lr)
            
            # Calculate metrics
            psnr = calculate_psnr(sr, hr)
            ssim = calculate_ssim(sr, hr)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values)
    } 