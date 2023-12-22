import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if d >= temporal_unit:
            loss += temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    return loss / d

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    positive_pairs = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(positive_pairs, positive_pairs.transpose(1, 2))  # B x 2T x 2T
    
    positive_logits = sim[:, :T, T:]  
    positive_logits = -F.log_softmax(positive_logits, dim=-1)
    
    loss = positive_logits.mean()
    return loss
