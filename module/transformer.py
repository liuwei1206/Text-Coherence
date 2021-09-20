# author = liuwei
# date = 2021-09-20

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
import torch.nn.TransformerEncoderLayer as TransformerEncoderLayer
import torch.nn.TransformerEncoder as TransformerEncoder
import numpy as np

class Transformer(nn.Module):
    """
    This is an encapsulated Transformer Encoder
    """
    def __init__(self, d_model, nhead, num_layers):
        """
        Args:
            d_model: dimension of the model
            nhead: number of head
            num_layers: the number of transformer encoder layer
        """
        super(Transformer, self).__init__()

        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_layers
        )

        self.d_model = d_model
        self.nhead= nhead
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_mask=None):
        """
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        output = self.transformer_encoder(src, mask, src_key_mask)

        return output