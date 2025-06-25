import torch
import torch.nn as nn
from transformers import RobertaModel
from vit import ViT

class TextEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, projection_dim)
        )
        
        #freeze first 6 layers
        for i, layer in enumerate(self.model.encoder.layer):
            if i < 6:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_token)

class ImageEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.model = ViT(
            img_height=224,
            img_width=224,
            patch_size=16,
            projection_dims=projection_dim
        )
        
    def forward(self, images):
        return self.model(images)

class CLIP(nn.Module):
    def __init__(self, projection_dim=256, temperature=0.07):
        super().__init__()
        self.text_encoder = TextEncoder(projection_dim)
        self.image_encoder = ImageEncoder(projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(temperature).exp())
        
    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        #normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        #cosine similarity
        logit_scale = self.logit_scale.exp()
        logits = (text_features @ image_features.t()) * logit_scale
        
        return logits