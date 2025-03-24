# Copyright (c) Duowang Zhu.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ChangeDecoder(nn.Module):
    """
    Decoder network that upsamples feature maps and produces final predictions.
    """
    
    def __init__(self, args, in_dim: List[int] = [64, 128, 256, 384], has_sigmoid: bool = False) -> None:
        """
        Initialize the decoder network.
        
        Args:
            in_dim: Dimensions of input feature maps from encoder stages
            num_class: Number of output channels in final prediction
        """
        super().__init__()
        self.has_sigmoid = has_sigmoid
        
        # Extract channel dimensions from input
        c1_channel, c2_channel, c3_channel, c4_channel = in_dim
        
        # Upsampling block for c4 features
        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, c3_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c3_channel, c3_channel, kernel_size=4, stride=2, padding=1)
        )
        
        # Upsampling block for c3 features
        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, c2_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1)
        )
        
        # Upsampling block for c2 features
        self.up_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, c1_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c1_channel, c1_channel, kernel_size=4, stride=2, padding=1)
        )
        
        # Final prediction layer
        if self.has_sigmoid:
            num_class = 1
        else:
            num_class = args.num_class

        self.up_c1 = nn.Sequential(
            nn.Conv2d(c1_channel, num_class, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, f: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            f: List of feature maps from encoder [c1, c2, c3, c4]
                
        Returns:
            Predicted output tensor
        """
        # Unpack feature maps
        c1, c2, c3, c4 = f
        
        # Progressive upsampling with skip connections
        c3f = c3 + self.up_c4(c4)
        c2f = c2 + self.up_c3(c3f)
        c1f = c1 + self.up_c2(c2f)
        
        # Final prediction
        pred = self.up_c1(c1f)

        if self.has_sigmoid:
            pred = torch.sigmoid(pred)
        
        return pred