import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import os
from typing import Optional, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class LoRATrainer:
    """trainer class for LoRA fine-tuning"""
    
    def __init__(self, model, config, accelerator: Accelerator):
        self.model = model
        self.config = config
        self.accelerator = accelerator
        
        #initialize optimizer (only LoRA parameters)
        lora_params = model.get_lora_parameters()
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        #initialize scheduler
        self.scheduler = None
        
        #tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        #print model info
        if accelerator.is_main_process:
            model.print_trainable_parameters()
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """main training loop."""
        #calculate total steps
        total_steps = self.config.max_steps
        
        #initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        #distributed training
        self.model, self.optimizer, train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, self.scheduler
        )
        
        if val_loader:
            val_loader = self.accelerator.prepare(val_loader)
        
        #training loop
        self.model.train()
        train_iterator = iter(train_loader)
        
        progress_bar = tqdm(
            range(total_steps),
            desc="Training",
            disable=not self.accelerator.is_main_process
        )
        
        for step in progress_bar:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            #forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            #backward pass
            self.accelerator.backward(loss)
            
            #gradient clipping
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            
            #optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            #logging
            if self.global_step % self.config.logging_steps == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/global_step": self.global_step
                }
                
                if self.accelerator.is_main_process:
                    if wandb.run is not None:
                        wandb.log(metrics)
                    
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
            
            #eval
            if val_loader and self.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate(val_loader)
                
                if self.accelerator.is_main_process:
                    if wandb.run is not None:
                        wandb.log(eval_metrics)
                    
                    if eval_metrics["val/loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val/loss"]
                        self.save_model(os.path.join(self.config.output_dir, "best_model"))
                
                self.model.train()  
            
            #save checkpoint
            if self.global_step % self.config.save_steps == 0:
                if self.accelerator.is_main_process:
                    self.save_model(os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}"))
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """evaluate the model on validation set"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="evaluating", disable=not self.accelerator.is_main_process):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
            "val/global_step": self.global_step
        }
        
        if self.accelerator.is_main_process:
            print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def save_model(self, output_dir: str):
        """Save the model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        
        #save LoRA weights
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_lora_weights(os.path.join(output_dir, "lora_weights.pt"))
        
        #save training state
        training_state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config
        }
        
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
        
        print(f"Model saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """load a checkpoint"""
        #load LoRA weights
        lora_weights_path = os.path.join(checkpoint_dir, "lora_weights.pt")
        if os.path.exists(lora_weights_path):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_lora_weights(lora_weights_path)
        
        #load training state
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location='cpu')
            self.global_step = training_state["global_step"]
            self.best_val_loss = training_state["best_val_loss"]
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            
            if self.scheduler and training_state["scheduler_state_dict"]:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
        
        print(f"checkpoint loaded from {checkpoint_dir}")