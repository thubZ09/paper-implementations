import torch
import torch.nn.functional as F

def siglip_loss(logits, labels):
    """sigmoid loss for pairwise similarity"""
    return -torch.sum(F.logsigmoid(logits * labels)) / logits.size(0)

def create_labels(batch_size, device):
    """to create target labels (diagonal = positive)"""
    return 2 * torch.eye(batch_size, device=device) - torch.ones(batch_size, device=device)