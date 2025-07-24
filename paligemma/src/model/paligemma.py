import torch
from torch import nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig

class PaliGemmaModel(nn.Module):
    """creates a PaliGemma model for vision-language tasks.
    
    this class wraps the HuggingFace PaliGemma model and provides
    additional functionality for fine-tuning and inference.
    
    Args:
        model_name: String name of the PaliGemma model to load.
        num_classes: Number of output classes (if doing classification).
        freeze_backbone: Whether to freeze the backbone parameters.
    """
    
    def __init__(self, 
                 model_name: str = "google/paligemma-3b-pt-224",
                 num_classes: int = None,
                 freeze_backbone: bool = False) -> None:
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes

        try:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to creating model from config...")
            config = PaliGemmaConfig.from_pretrained(model_name)
            self.model = PaliGemmaForConditionalGeneration(config)
        
        #freeze backbone 
        if freeze_backbone:
            self.freeze_backbone_parameters()
        
        #add classification head 
        if num_classes:
            self.classifier = nn.Linear(
                self.model.config.text_config.hidden_size,
                num_classes
            )
        else:
            self.classifier = None
    
    def freeze_backbone_parameters(self):
        """Freeze the backbone parameters to prevent updates during training."""
        for param in self.model.vision_tower.parameters():
            param.requires_grad = False
        
        #optional
        language_layers = self.model.language_model.model.layers
        for layer in language_layers[:-2]:  
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, inputs):
        """forward pass through the PaliGemma model.
        
        Args:
            inputs: Dictionary containing 'input_ids', 'attention_mask', 'pixel_values'
            
        Returns:
            Model outputs (logits for text generation or classification)
        """
        if isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs)

        if self.classifier is not None:
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
            pooled_output = hidden_states.mean(dim=1)
            outputs.logits = self.classifier(pooled_output)
        
        return outputs
    
    def generate(self, inputs, **generation_kwargs):
        """generate text given image and text inputs.
        
        Args:
            inputs: Dictionary containing model inputs
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Generated token sequences
        """
        return self.model.generate(**inputs, **generation_kwargs)
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

def create_paligemma_model(model_name: str = "google/paligemma-3b-pt-224",
                          num_classes: int = None,
                          freeze_backbone: bool = False):
    """actory function to create a PaliGemma model.
    
    Args:
        model_name: HF model name or path
        num_classes: number of classes for classification (none for generation)
        freeze_backbone: Whether to freeze backbone parameters
        
    Returns:
        PaliGemmaModel instance
    """
    model = PaliGemmaModel(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )
    
    print(f"Created PaliGemma model: {model_name}")
    print(f"Total parameters: {model.get_total_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    return model

#for backward compatibility, keep a simple class similar to TinyVGG structure
class SimplePaliGemma(nn.Module):
    """Simplified PaliGemma wrapper for basic usage."""
    
    def __init__(self, model_name: str = "google/paligemma-3b-pt-224"):
        super().__init__()
        self.paligemma = create_paligemma_model(model_name)
    
    def forward(self, x):
        return self.paligemma(x)