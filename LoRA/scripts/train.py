"""
training script for LoRA fine-tuning on Colab T4.

"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import argparse
from config.config import ModelConfig
from src.data.dataset import IMDBDataset, AlpacaDataset
from src.models.lora_model import LoRAGPT2
from src.training.trainer import LoRATrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.py")
    parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "alpaca"])
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    args = parser.parse_args()
    
    #load config
    config = ModelConfig()
    
    #onitialize accelerator for mixed precision training
    accelerator = Accelerator(
        mixed_precision="fp16" if config.mixed_precision else "no",
        gradient_accumulation_steps=1,
        log_with="wandb" if args.wandb else None,
        project_dir=config.output_dir,
    )
    
    #initialize wandb
    if args.wandb and accelerator.is_main_process:
        wandb.init(
            project="lora-gpt2-finetuning",
            config=config.__dict__,
            name=f"lora-{args.dataset}-rank{config.rank}"
        )
    
    #load datasets
    if args.dataset == "imdb":
        train_dataset = IMDBDataset("train", config.model_name, config.max_length, config.max_train_samples)
        val_dataset = IMDBDataset("test", config.model_name, config.max_length, config.max_val_samples)
    else:
        train_dataset = AlpacaDataset("train", config.model_name, config.max_length, config.max_train_samples)
        val_dataset = AlpacaDataset("val", config.model_name, config.max_length, config.max_val_samples)
    
    #create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    #initialize model
    model = LoRAGPT2(config)
    
    #initialize nd train 
    trainer = LoRATrainer(model, config, accelerator)
    trainer.train(train_loader, val_loader)

    if accelerator.is_main_process:
        trainer.save_model(os.path.join(config.output_dir, "final_model"))
        print("Training completed!")

if __name__ == "__main__":
    main()