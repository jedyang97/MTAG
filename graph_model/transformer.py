import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        encoder_norm = nn.LayerNorm(ninp)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.ninp = ninp

    def forward(self, src):
        r"""
        Shape:
            - src: :math:`(S, N, E)`.
            - output: :math:`(S, N, E)`.
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Added to support odd d_model
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)