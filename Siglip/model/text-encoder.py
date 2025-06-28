from transformers import RobertaModel, RobertaTokenizer
import torch.nn as nn
from configs.base_config import BaseConfig

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BaseConfig()
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.text_encoder)
        self.model = RobertaModel.from_pretrained(self.config.text_encoder)
        self.projection = nn.Linear(
            self.model.config.hidden_size, 
            self.config.projection_dim
        )
        self._freeze_layers()
    
    def _freeze_layers(self):
        #freeze first 6 layers
        for i, layer in enumerate(self.model.encoder.layer[:6]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = outputs.last_hidden_state[:, 0, :]
        return self.projection(embeddings)