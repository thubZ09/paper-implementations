#src/model/lora_layers.py
import torch
import torch.nn as nn
from typing import Optional

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        #LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        #initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling

class LoRALinear(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear_layer
        self.linear.requires_grad_(False)  # Freeze original weights
        
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha,
            dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)