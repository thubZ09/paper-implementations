import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from configs.base_config import BaseConfig

class SigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BaseConfig()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.temperature = nn.Parameter(torch.tensor(self.config.temperature))
        self.bias = nn.Parameter(torch.tensor(self.config.bias))
        self.ln = nn.LayerNorm(self.config.projection_dim)
        
    def forward(self, images, input_ids, attention_mask):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(input_ids, attention_mask)
        
        #ormalize embeddings
        img_emb = F.normalize(self.ln(img_emb), dim=-1)
        txt_emb = F.normalize(self.ln(txt_emb), dim=-1)
        
        #similarity matrix
        logits = torch.einsum("i d, j d -> i j", txt_emb, img_emb)
        return -logits * self.temperature + self.bias
    
    def encode_image(self, image):
        with torch.no_grad():
            return self.image_encoder(image)
    
    def encode_text(self, texts):
        with torch.no_grad():
            tokens = self.text_encoder.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
            return self.text_encoder(
                tokens["input_ids"].to(self.config.device), 
                tokens["attention_mask"].to(self.config.device)
            )