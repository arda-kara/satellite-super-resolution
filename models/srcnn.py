import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    Super-Resolution CNN (SRCNN)
    
    A simpler and faster alternative to ESRGAN, but with lower quality results.
    The model consists of three convolutional layers:
    1. Feature extraction
    2. Non-linear mapping
    3. Reconstruction
    
    Reference: Dong et al. "Learning a Deep Convolutional Network for Image Super-Resolution"
    """
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction layer
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Note: SRCNN expects the input to be already upscaled to the target resolution
        using bicubic interpolation. This is different from ESRGAN which does the
        upscaling within the network.
        """
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out


class FSRCNN(nn.Module):
    """
    Fast Super-Resolution CNN (FSRCNN)
    
    An improved version of SRCNN that is faster and more efficient.
    The model consists of:
    1. Feature extraction
    2. Shrinking
    3. Multiple mapping layers
    4. Expanding
    5. Deconvolution (for upscaling)
    
    Reference: Dong et al. "Accelerating the Super-Resolution Convolutional Neural Network"
    """
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        
        # Feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        
        # Shrinking
        self.mid_part = [
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        ]
        
        # Mapping (m mapping layers)
        for _ in range(m):
            self.mid_part.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s)
            ])
            
        # Expanding
        self.mid_part.extend([
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        ])
        
        self.mid_part = nn.Sequential(*self.mid_part)
        
        # Deconvolution (upscaling)
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9, stride=scale_factor, 
            padding=4, output_padding=scale_factor-1
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x 