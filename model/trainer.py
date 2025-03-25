# Copyright (c) Duowang Zhu.
# All rights reserved.

from functools import partial
import math
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat

from model.x3d import create_x3d
from model.change_decoder import ChangeDecoder
from model.caption_decoder import CaptionDecoder
from model.utils import weight_init


class Encoder(nn.Module):
    """
    Encoder model based on X3D architecture with feature enhancement capabilities.
    
    This encoder processes video frames using X3D architecture and enhances
    intermediate features using learnable tokens and temporal differences.
    """
    
    def __init__(self, args: Any, embed_dims: List[int]) -> None:
        """
        Initialize the encoder.
        
        Args:
            args: Configuration arguments
            embed_dims: Dimensions of embeddings at each stage
        """
        super().__init__()
        self.args = args

        # Initialize X3D backbone
        self.x3d = create_x3d(input_clip_length=3, depth_factor=5.0)

        # Load pretrained weights
        try:
            state_dict = torch.load(args.pretrained, map_location='cpu')['model_state']
            msg = self.x3d.load_state_dict(state_dict, strict=True)
            print(f'Load pretrained weight: {args.pretrained}, {msg}.')
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

        # Learnable perception frames for change extrcation
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, args.num_perception_frame, args.in_height, args.in_width), 
            requires_grad=True
        )

        # Feature enhancement layers for each stage
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    dim, 
                    dim, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    bias=False
                ),
                nn.ReLU()
            ) for dim in embed_dims
        ])

    def enhance(self, x: torch.Tensor, fc: nn.Module) -> torch.Tensor:
        """
        Enhance the binary change frame using temporal information from pre/post frames.
        
        Args:
            x: Input tensor with shape [B, C, T, H, W] where T is the temporal dimension
            fc: Feature enhancement module for refining temporal features

        Returns:
            torch.Tensor: Enhanced tensor with same shape as input, with middle
                        frame enhanced using temporal information

        Note:
            The enhancement is only applied to the middle frame (T//2), leaving
            other frames unchanged.
        """
        # Extract tensor dimensions
        middle_idx = x.shape[2] // 2
        
        # Get the pre and post frames based on configuration
        pre_frame = x[:, :, 0]  # First frame
        post_frame = x[:, :, self.args.num_perception_frame+1]  # Specified post frame
        
        # Calculate temporal difference features
        temporal_diff = torch.abs(pre_frame - post_frame)
        
        # Apply feature enhancement module to extract refinement features
        enhancement_features = fc(temporal_diff)
        
        # Apply enhancement to middle frame only (residual enhancement)
        middle_frame = x[:, :, middle_idx]
        enhanced_middle_frame = middle_frame + enhancement_features
        
        # Create output by copying input and replacing middle frame
        enhanced_x = x.clone()
        enhanced_x[:, :, middle_idx] = enhanced_middle_frame
        
        return enhanced_x

    def base_forward(self, x: torch.Tensor, output_final: bool=False) -> List[torch.Tensor]:
        """
        Forward pass through X3D blocks with enhancement.
        
        Args:
            x: Input tensor with shape [B, C, T, H, W]
            
        Returns:
            List of feature maps from different stages
        """
        if output_final:
            for i in range(5):
                x = self.x3d.blocks[i](x)

            return x[:, :, self.args.num_perception_frame]
        else:
            out = []
            # Process through X3D blocks with enhancement
            for i in range(4):
                # Process through X3D block
                x = self.x3d.blocks[i](x)

                # Apply enhancement
                x = self.enhance(x, self.fc[i])

                # Extract middle frame features
                layer_feature = []
                for idx in range(self.args.num_perception_frame):
                    layer_feature.append(x[:, :, idx+1])
                out.append(layer_feature)

            return out

    def forward(self, x: torch.Tensor, y: torch.Tensor, output_final: bool=False) -> List[torch.Tensor]:
        """
        Forward pass with input and target frames.
        
        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]
            
        Returns:
            List of feature tensors from different stages
        """        
        # Expand tokens to match batch size
        expand_percep_frames = repeat(self.perception_frames, '1 c t h w -> b c t h w', b=x.shape[0])

        # Combine into 3-frame sequence [input, token, target]
        frames = torch.cat([
            x.unsqueeze(2),
            expand_percep_frames,
            y.unsqueeze(2)
        ], dim=2)

        # Process through network
        features = self.base_forward(frames, output_final)

        return features


class Trainer(nn.Module):
    """
    Complete model with encoder and decoder for video frame enhancement.
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize the trainer with encoder and decoder.
        
        Args:
            args: Configuration arguments
        """
        super().__init__()
        self.args = args
    
        # Define embedding dimensions for each stage
        self.embed_dims = [24, 24, 48, 96]
        
        # Initialize encoder and decoder
        self.encoder = Encoder(args, self.embed_dims)
        
        # For binary change detection and change caption task
        if args.num_perception_frame == 1 and 'CD' in args.dataset:
            self.decoder = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
            # Initialize decoder weights
            weight_init(self.decoder)

        # For semantic change detection task
        elif args.num_perception_frame == 3:
            self.decoder_pre = ChangeDecoder(args, in_dim=self.embed_dims)
            self.decoder_post = ChangeDecoder(args, in_dim=self.embed_dims)
            self.decoder_change = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
            # Initialize decoder weights
            weight_init(self.decoder_pre)
            weight_init(self.decoder_post)
            weight_init(self.decoder_change)

        # For building damage assessment task
        elif args.num_perception_frame == 2:
            self.decoder_cls = ChangeDecoder(args, in_dim=self.embed_dims)
            self.decoder_loc = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
            # Initialize decoder weights
            weight_init(self.decoder_cls)
            weight_init(self.decoder_loc)

        # For change caption task
        elif args.num_perception_frame == 1 and 'CC' in args.dataset:
            self.decoder = CaptionDecoder(args)
        else:
            assert False

    def update_bcd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y)

        # perception feature
        perception_change_feat = list(map(lambda x: x[0], features))

        # Generate prediction using decoder
        prediction = self.decoder(perception_change_feat)

        return prediction
    
    def update_scd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y)
        
        # Generate prediction using decoder
        perception_pre_feat = list(map(lambda x: x[0], features))
        perception_change_feat = list(map(lambda x: x[1], features))
        perception_post_feat = list(map(lambda x: x[2], features))

        pre_mask = self.decoder_pre(perception_pre_feat)
        post_mask = self.decoder_post(perception_post_feat)
        change_mask = self.decoder_change(perception_change_feat)

        return pre_mask, post_mask, change_mask
    
    def update_bda(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y)

        # perception feature
        perception_cls_feat = list(map(lambda x: x[0], features))
        perception_loc_feat = list(map(lambda x: x[1], features))

        # Generate prediction using decoder
        pred_cls = self.decoder_cls(perception_cls_feat)
        pred_loc = self.decoder_loc(perception_loc_feat)

        return pred_cls, pred_loc
    
    def update_cc(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y, output_final=True)

        return features
