import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .dataset import get_dataset
from .augmentations import BasicAugment
from .scheduler import get_cosine_schedule_with_warmup
from models.vit import VisionTransformer
from configs.vit_small import ViT_Small as config

def train_vit():
    #config
    cfg = config()
    
    #dataset nd dataloader
    train_dataset, _ = get_dataset('cifar10', cfg.image_size, train=True)
    val_dataset, _ = get_dataset('cifar10', cfg.image_size, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    model = VisionTransformer(cfg).to(cfg.device)
    
    #optimizer nd scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps
    )
    
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #mixed precision
    scaler = GradScaler(enabled=cfg.mixed_precision)
    
    #training loop
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for images, labels in progress:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            
            #mixed precision forward
            with autocast(enabled=cfg.mixed_precision):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            #gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            #update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            #meetrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress.set_postfix(
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr'],
                acc=100.*correct/total
            )
        
        #validation
        val_loss, val_acc = validate(model, val_loader, criterion, cfg)
        print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        #checkpoint
        torch.save(model.state_dict(), f"vit_epoch_{epoch+1}.pth")

def validate(model, val_loader, criterion, cfg):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            
            with autocast(enabled=cfg.mixed_precision):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

if __name__ == "__main__":
    train_vit()