import torch
from torch.nn import functional as F


class SmoothCrossEntropyLoss(torch.nn.Module):

    def __init__(self, tgt_pad_idx, eps=0.4, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.tgt_pad_idx = tgt_pad_idx
        self.eps = eps

        assert reduction in ['mean', 'sum'], 'reduction should be mean or sum'
        self.reduction = reduction

    def forward(self, pred, tgt):
        pred = torch.flatten(pred, start_dim=0, end_dim=1)
        tgt = torch.flatten(tgt, start_dim=0, end_dim=1)

        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred)
        one_hot = one_hot.scatter(1, tgt.view(-1, 1), 1)
        one_hot_eps = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        nll = -(one_hot_eps * log_prob).sum(dim=1)
        mask = tgt.ne(self.tgt_pad_idx)

        if self.reduction == 'mean':
            return torch.mean(nll.masked_select(mask))

        return torch.sum(nll.masked_select(mask))
