import numpy as np
import torch


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = ~(torch.from_numpy(mask) == 0).squeeze(0)
    matrix_ninf = torch.ones(()) * float('-inf')
    matrix_zeros = torch.zeros(()).float()
    mask = torch.where(mask, matrix_ninf, matrix_zeros)
    return mask
