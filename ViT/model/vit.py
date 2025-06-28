import torch
import torch.nn as nn
from .embeddings import PatchEmbedding
from .attention import Attention
from .mlp import MLP
from .utils import PreNorm, Residual
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attention = Residual(
            PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))
        )
        self.mlp = Residual(
            PreNorm(dim, MLP(dim, mlp_dim, dropout=dropout))
        )
    
    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            channels=config.channels,
            dim=config.dim
        )
        
        #transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout
            ) for _ in range(config.depth)
        ])
        
        #classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embed(x)
        
        for block in self.blocks:
            if self.training and self.config.mixed_precision:
                x = checkpoint(block, x)  #gradient checkpointing
            else:
                x = block(x)
        
        cls_token = x[:, 0]
        return self.mlp_head(cls_token)
    
    def extract_features(self, x):
        x = self.patch_embed(x)
        features = []
        
        for block in self.blocks:
            x = block(x)
            features.append(x[:, 0])  
        
        return features