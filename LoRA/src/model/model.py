import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Any
from .lora_layers import LoRALinear

class LoRAGPT2(nn.Module):
    """GPT-2 model with LoRA adapters applied to attention layers"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #pre-trained GPT-2
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
            device_map=None  
        )
        
        #freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        #apply LoRA to attention layers
        self.apply_lora()
        
        #enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
    
    def apply_lora(self):
        """apply LoRA adapters to attention projection layers"""
        target_modules = ["c_attn", "c_proj"] 
        
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    #replace with LoRA linear layer
                    lora_layer = LoRALinear(
                        module,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout
                    )
                    
                    #replace the module
                    parent_name = name.rsplit('.', 1)[0]
                    child_name = name.rsplit('.', 1)[1]
                    parent_module = dict(self.base_model.named_modules())[parent_name]
                    setattr(parent_module, child_name, lora_layer)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """forward pass through the model"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate(self, input_ids, **kwargs):
        """generate text using the model"""
        return self.base_model.generate(input_ids, **kwargs)
    
    def get_lora_parameters(self):
        """get only the LoRA parameters for optimization"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params.append(param)
        return lora_params
    
    def save_lora_weights(self, path: str):
        """save only the LoRA weights"""
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_state_dict[name] = param.cpu()
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'config': self.config
        }, path)
    
    def load_lora_weights(self, path: str):
        """load LoRA weights"""
        checkpoint = torch.load(path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        #only LoRA parameters
        model_dict = self.state_dict()
        lora_dict = {k: v for k, v in lora_state_dict.items() if k in model_dict}
        model_dict.update(lora_dict)
        self.load_state_dict(model_dict, strict=False)
    
    def print_trainable_parameters(self):
        """print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,}")
        print(f"All params: {all_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")