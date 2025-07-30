import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding1D(nn.Module):
    """PE function."""

    def __init__(self, d_model, dropout=0.1, max_len=1000, positional_scaler=1.0):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) * positional_scaler
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        shape of x has to be: (N, S, F)   # N = batch size, S = sequence length, F = feature dimension
        """
        pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + pe
        return self.dropout(x)


class A2DPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, positional_scaler=1.0):
        super(A2DPE, self).__init__()
        self.alpha_fc = PEScaling()
        self.beta_fc = PEScaling()
        pe = PositionalEncoding1D(d_model=d_model, dropout=dropout, positional_scaler=positional_scaler).pe.squeeze()
        self.pe = pe

    def forward(self, x):
        pe = torch.stack([self.pe] * x.shape[0]).to(x.device)
        alpha, beta = self.alpha_fc(x), self.beta_fc(x)
        ph = alpha * pe[:, :x.shape[-2]]
        pw = beta * pe[:, :x.shape[-1]]
        ph = torch.repeat_interleave(ph, x.shape[-1], dim=1)
        pw = pw.repeat(1, x.shape[-2], 1)
        pos = (ph + pw).reshape(x.shape)
        x = pos + x
        return x


class PEScaling(nn.Module):
    def __init__(self):
        super(PEScaling, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        # global avg pooling
        e = x.mean(dim=(-1, -2, -3)).unsqueeze(-1)
        return self.sigmoid(self.linear2(self.relu(self.linear1(e)))).unsqueeze(-1)
