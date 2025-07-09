import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoTokenizer
import wandb

def set_seed(seed: int = 42):
    """set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model: nn.Module) -> Dict[str, float]:
    """get model size"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": size_mb
    }

def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """estimate FLOPs for a forward pass"""
    #this is a simplified FLOP calculation
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            output_dims = output.shape[2:]
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            filters_per_channel = out_channels // module.groups
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // module.groups
            active_elements_count = int(np.prod(output_dims))
            total_flops += conv_per_position_flops * active_elements_count * filters_per_channel
    
    #register hooks
    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(flop_count_hook))
    
    #forward pass
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        model(dummy_input)
    
    #remove hooks
    for handle in handles:
        handle.remove()
    
    return total_flops

def format_time(seconds: float) -> str:
    """format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def compute_metrics(eval_pred) -> Dict[str, float]:
    """compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # This is a simplified version - adjust based on your needs
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    #flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    #filter out padding tokens (assuming -100 is used for padding)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    #calculate perplexity
    loss = nn.CrossEntropyLoss()(predictions, labels)
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
        "eval_loss": loss.item()
    }

def save_metrics(metrics: Dict[str, Any], output_dir: str, filename: str = "metrics.json"):
    """Save metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float] = None,
    output_dir: str = None,
    show_plot: bool = True
):
    """plot training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    #plot perplexity
    plt.subplot(1, 2, 2)
    train_perplexity = [np.exp(loss) for loss in train_losses]
    plt.plot(train_perplexity, label='Training Perplexity', color='blue')
    if val_losses:
        val_perplexity = [np.exp(loss) for loss in val_losses]
        plt.plot(val_perplexity, label='Validation Perplexity', color='orange')
    plt.title('Training Perplexity')
    plt.xlabel('Steps')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_learning_rate_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_schedule_type: str = "cosine"
):
    """create learning rate schedule."""
    from transformers import get_scheduler
    
    return get_scheduler(
        name=lr_schedule_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def log_gpu_memory():
    """;og GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        memory_info = {
            "gpu_memory_allocated_gb": allocated,
            "gpu_memory_cached_gb": cached,
            "gpu_memory_max_allocated_gb": max_allocated
        }
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB")
        
        #log to wandb if available
        if wandb.run is not None:
            wandb.log(memory_info)
        
        return memory_info
    else:
        return {"gpu_memory_allocated_gb": 0, "gpu_memory_cached_gb": 0, "gpu_memory_max_allocated_gb": 0}

def cleanup_checkpoints(output_dir: str, keep_last_n: int = 3):
    """clean up old checkpoints, keeping only the last n."""
    checkpoint_dirs = []
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoint_dirs.append((step, item_path))
            except ValueError:
                continue
    
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    if len(checkpoint_dirs) > keep_last_n:
        for step, checkpoint_path in checkpoint_dirs[:-keep_last_n]:
            print(f"Removing old checkpoint: