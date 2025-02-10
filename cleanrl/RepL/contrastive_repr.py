"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConLoss(nn.Module):
    def __init__(self):
        super(SpectralConLoss, self).__init__()

    def forward(self, data: torch.Tensor, next_data: torch.Tensor):
        X = data
        Y = next_data
        # Credits https://github.com/jhaochenz96/spectral_contrastive_learning/blob/ee431bdba9bb62ad00a7e55792213ee37712784c/models/spectral.py#L8C1-L17C96
        assert X.shape == Y.shape
        assert X.ndim == 2

        npts, dim = X.shape
        diag = 2 * torch.mean(X * Y) * dim
        square_term = torch.matmul(X, Y.T) ** 2
        off_diag = -(
            torch.mean(
                torch.triu(square_term, diagonal=1)
                + torch.tril(square_term, diagonal=-1)
            )
            * npts
            / (npts - 1)
        )
        score = diag + off_diag
        return -score

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, data, next_data, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            data: hidden vector of shape [bsz, n_views, ...].
            next_data: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.stack([data, next_data], dim=1)
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        print(loss, "loss")

        return loss

class NoiseConLoss(nn.Module):
    """Making Linear MDPs Practical via Contrastive Representation Learning: https://https://arxiv.org/pdf/2207.07150.
    """
    def __init__(self, device):
        super(NoiseConLoss, self).__init__()
        self.device = device

    def forward(self, data: torch.Tensor, next_data: torch.Tensor):
        labels = torch.eye(data.shape[0]).to(self.device)

        # we take NCE gamma = 1 here, the paper uses 0.2
        contrastive = (data[:, None, :] * next_data[None, :, :]).sum(-1) 
        model_loss = nn.CrossEntropyLoss()
        model_loss = model_loss(contrastive, labels)
        return model_loss
    

class infoNCELoss(nn.Module):
    """Representation Learning with Contrastive Predictive Coding: https://arxiv.org/abs/1807.03748.
    """
    def __init__(self, device, temperature=1):
        super(infoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, data: torch.Tensor, next_data: torch.Tensor):
        # Positive keys are the entries on the diagonal
        labels = torch.arange(data.shape[0]).to(self.device)

        logits = data @ next_data.T

        return F.cross_entropy(logits/self.temperature, labels, reduction='mean')