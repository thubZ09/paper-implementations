from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    #model param
    model_name: str = "gpt2"
    rank: int = 8  
    alpha: int = 16  
    dropout: float = 0.1
    
    #training (for T4)
    batch_size: int = 8  
    max_length: int = 512  
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    
    #sys param
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  
    gradient_checkpointing: bool = True  
    
    #data param
    dataset_name: str = "imdb"  
    max_train_samples: int = 10000 
    max_val_samples: int = 1000
    
    #eval
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    
    #paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"