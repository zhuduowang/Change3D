import torch
import os
import math
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_
from typing import Optional
from torch.nn import functional as F
from model.utils import weight_init


class CrossTransformer(nn.Module):
    """
    Cross Transformer layer for feature interaction between image pairs.
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        Initialize the Cross Transformer layer.
        
        Args:
            dropout (float): Dropout rate.
            d_model (int): Dimension of hidden state.
            n_head (int): Number of heads in multi-head attention.
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        """
        Forward pass of Cross Transformer.
        
        Args:
            input1 (Tensor): First input tensor.
            input2 (Tensor): Second input tensor.
            
        Returns:
            tuple: Transformed outputs for both inputs.
        """
        # Compute difference as the key-value pairs
        dif = input2 - input1
        output_1 = self.cross(input1, dif)  # (Q,K,V)
        output_2 = self.cross(input2, dif)  # (Q,K,V)

        return output_1, output_2
    
    def cross(self, input, dif):
        """
        Cross-attention mechanism with difference as key and value.
        
        Args:
            input (Tensor): Query tensor.
            dif (Tensor): Key-value tensor (difference).
            
        Returns:
            Tensor: Transformed output.
        """
        # RSICCformer_D (diff_as_kv)
        attn_output, _ = self.attention(input, dif, dif)  # (Q,K,V)

        # First residual connection and normalization
        output = input + self.dropout1(attn_output)
        output = self.norm1(output)
        
        # Feed-forward network
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        
        # Second residual connection and normalization
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        
        return output


class ResBlock(nn.Module):
    """
    Residual Block for feature refinement.
    """
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        """
        Initialize the Residual Block.
        
        Args:
            in_channel (int): Input channel dimension.
            out_channel (int): Output channel dimension.
            stride (int): Stride for convolutions.
            shortcut (nn.Module, optional): Shortcut connection.
        """
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, int(out_channel/2), kernel_size=1),
            nn.BatchNorm2d(int(out_channel/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channel/2), int(out_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channel/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channel/2), out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        """
        Forward pass of Residual Block.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after residual connection.
        """
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)


class MCCFormers_diff_as_Q(nn.Module):
    """
    Multi-modal Change Captioning Transformer with difference as query.
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        """
        Initialize the MCCFormers model.
        
        Args:
            feature_dim (int): Dimension of input features.
            dropout (float): Dropout rate.
            h (int): Height of feature maps.
            w (int): Width of feature maps.
            d_model (int): Dimension of hidden state.
            n_head (int): Number of heads in multi-head attention.
            n_layers (int): Number of transformer layers.
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        print(f"encoder_n_layers={n_layers}")

        # Position embeddings
        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h*w, int(d_model))

        # Feature projection layers for different input dimensions
        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)

        # Transformer layers
        self.transformer = nn.ModuleList([
            CrossTransformer(dropout, d_model, n_head) for _ in range(n_layers)
        ])

        # Residual blocks for feature fusion
        self.resblock = nn.ModuleList([
            ResBlock(d_model*2, d_model*2) for _ in range(n_layers)
        ])

        # Layer normalization for each transformer output
        self.LN = nn.ModuleList([
            nn.LayerNorm(d_model*2) for _ in range(n_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        """
        Forward pass of the MCCFormers model.
        
        Args:
            img_feat1 (Tensor): Features from the first image.
            img_feat2 (Tensor): Features from the second image.
            
        Returns:
            Tensor: Fused output features.
        """
        # Get batch size and feature dimensions
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)

        # Apply appropriate projection based on input feature dimension
        if feature_dim == 1024:
            img_feat1 = self.projection(img_feat1)  
            img_feat2 = self.projection(img_feat2)  
        elif feature_dim == 768:
            img_feat1 = self.projection2(img_feat1)  
            img_feat2 = self.projection2(img_feat2)  
        elif feature_dim == 512:
            img_feat1 = self.projection3(img_feat1)  
            img_feat2 = self.projection3(img_feat2)  
        elif feature_dim == 256:
            img_feat1 = self.projection4(img_feat1)  
            img_feat2 = self.projection4(img_feat2)  

        # Generate positional embeddings
        pos_w = torch.arange(w).cuda()
        pos_h = torch.arange(h).cuda()
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        
        # Combine width and height embeddings
        position_embedding = torch.cat([
            embed_w.unsqueeze(0).repeat(h, 1, 1),
            embed_h.unsqueeze(1).repeat(1, w, 1)
        ], dim=-1)
        
        # Reshape for addition to feature maps
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)

        # Add positional embeddings to features
        img_feat1 = img_feat1 + position_embedding
        img_feat2 = img_feat2 + position_embedding

        # Reshape for transformer input
        encoder_output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)
        encoder_output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)

        # Process through transformer layers
        output1 = encoder_output1
        output2 = encoder_output2
        output1_list = []
        output2_list = []
        
        for layer in self.transformer:
            output1, output2 = layer(output1, output2)
            output1_list.append(output1)
            output2_list.append(output2)

        # Multi-scale feature fusion
        output = torch.zeros((196, batch, self.d_model*2)).cuda()
        
        for i, res in enumerate(self.resblock):
            # Concatenate outputs from both branches
            input_tensor = torch.cat([output1_list[i], output2_list[i]], dim=-1)
            output = output + input_tensor
            
            # Apply residual block
            output = output.permute(1, 2, 0).view(batch, self.d_model*2, 14, 14)
            output = res(output)
            
            # Reshape and normalize
            output = output.view(batch, self.d_model*2, -1).permute(2, 0, 1)
            output = self.LN[i](output)

        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        # Alternative learnable embedding
        self.embedding_1D = nn.Embedding(52, int(d_model))
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Input with positional encoding added.
        """
        # Fixed positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with multi-head attention mechanisms.
    """

    __constants__ = ['batch_first', 'norm_first']
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        """
        Initialize the transformer decoder layer.
        
        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of feedforward network.
            dropout (float): Dropout rate.
            layer_norm_eps (float): Layer normalization epsilon.
            batch_first (bool): If True, batch dimension is first.
            norm_first (bool): If True, normalization is applied before attention.
            device: Device to use.
            dtype: Data type to use.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        
        # Self-attention mechanisms
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention mechanisms
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        # Activation functions
        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        # Alpha blending layers
        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()
        weight_init(self)

    def init_weights(self):
        """
        Initialize weights for alpha blending.
        """
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer decoder layer.
        
        Args:
            tgt (Tensor): Target sequence.
            memory (Tensor): Memory from encoder.
            tgt_mask (Tensor, optional): Mask for target sequence.
            memory_mask (Tensor, optional): Mask for memory.
            tgt_key_padding_mask (Tensor, optional): Key padding mask for target.
            memory_key_padding_mask (Tensor, optional): Key padding mask for memory.
            
        Returns:
            Tensor: Output after self-attention and cross-attention.
        """
        # Self-attention block
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        
        # Cross-attention block
        enc_att, _ = self._mha_block2(
            (self_att_tgt),
            memory, memory_mask,
            memory_key_padding_mask
        )
        
        # Final normalization
        x = self.norm2(self_att_tgt + enc_att)
        
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                 key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Self-attention block.
        
        Args:
            x (Tensor): Input tensor.
            attn_mask (Tensor, optional): Attention mask.
            key_padding_mask (Tensor, optional): Key padding mask.
            
        Returns:
            Tensor: Output after self-attention.
        """
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> tuple:
        """
        Multi-head attention block 1.
        
        Args:
            x (Tensor): Query tensor.
            mem (Tensor): Key-value tensor.
            attn_mask (Tensor, optional): Attention mask.
            key_padding_mask (Tensor, optional): Key padding mask.
            
        Returns:
            tuple: (Output tensor, attention weights).
        """
        x, att_weight = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout2(x), att_weight
    
    def _mha_block2(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> tuple:
        """
        Multi-head attention block 2.
        
        Args:
            x (Tensor): Query tensor.
            mem (Tensor): Key-value tensor.
            attn_mask (Tensor, optional): Attention mask.
            key_padding_mask (Tensor, optional): Key padding mask.
            
        Returns:
            tuple: (Output tensor, attention weights).
        """
        x, att_weight = self.multihead_attn2(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout3(x), att_weight
    
    def _mha_block3(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> tuple:
        """
        Multi-head attention block 3.
        
        Args:
            x (Tensor): Query tensor.
            mem (Tensor): Key-value tensor.
            attn_mask (Tensor, optional): Attention mask.
            key_padding_mask (Tensor, optional): Key padding mask.
            
        Returns:
            tuple: (Output tensor, attention weights).
        """
        x, att_weight = self.multihead_attn3(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout4(x), att_weight

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Feed-forward block.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output after feed-forward network.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)


class CaptionDecoder(nn.Module):
    """
    Caption decoder for caption generation.
    """

    def __init__(self, args):
        """
        Initialize the caption decoder.
        
        Args:
            embed_dim (int): Dimension of input features.
            vocab_size (int): Size of vocabulary.
            n_head (int): Number of attention heads.
            n_layer (int): Number of decoder layers.
            dropout (float): Dropout rate.
        """
        super(CaptionDecoder, self).__init__()

        print(f"decoder_n_layers={args.n_layer}")

        # Embedding layer for vocabulary
        self.vocab_embedding = nn.Embedding(args.vocab_size, args.embed_dim)

        # Transformer decoder
        decoder_layer = Mesh_TransformerDecoderLayer(
            args.embed_dim, args.n_head,
            dim_feedforward=args.embed_dim * 4,
            dropout=args.dropout
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, args.n_layer)
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(args.embed_dim)

        # Output projection
        self.wdc = nn.Linear(args.embed_dim, args.vocab_size)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for better convergence.
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, memory, encoded_captions, caption_lengths):
        """
        Forward pass of the decoder.
        
        Args:
            memory (Tensor): Image features from encoder.
            encoded_captions (Tensor): Target captions.
            caption_lengths (Tensor): Length of each caption.
            
        Returns:
            tuple: (Predictions, sorted captions, decode lengths, sort indices).
        """
        # Transpose for sequence first
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)

        # Create mask for decoder self-attention
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()

        # Embed target sequence and add positional encoding
        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)

        # Pass through transformer decoder
        pred = self.transformer(tgt_embedding, memory, tgt_mask=mask)
        
        # Project to vocabulary size
        pred = self.wdc(self.dropout_layer(pred)).permute(1, 0, 2)

        # Sort by caption length for packing
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        
        # Get decode lengths for loss calculation (exclude <start>)
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind