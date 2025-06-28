import timm
import torch.nn as nn
from configs.base_config import BaseConfig

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BaseConfig()
        self.model = timm.create_model(
            self.config.image_encoder,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        self.projection = nn.Linear(
            self.model.num_features, 
            self.config.projection_dim
        )
        self._freeze_layers()
        
    def _freeze_layers(self):
        #freeze first 75% of layers
        total_params = len(list(self.model.parameters()))
        freeze_count = int(total_params * 0.75)
        
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i < freeze_count:
                param.requires_grad = False
    
    def forward(self, x):
        features = self.model(x)
        return self.projection(features)