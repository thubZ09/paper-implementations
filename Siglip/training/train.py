import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from training.dataset import ConceptualCaptionsDataset
from models.siglip import SigLIP
from training.losses import siglip_loss, create_labels
from configs.base_config import BaseConfig
from tqdm import tqdm
import torch.cuda.amp as amp

def train():
    config = BaseConfig()
    torch.manual_seed(config.seed)
    
    #initialize
    model = SigLIP().to(config.device)
    
    #datasets
    train_dataset = ConceptualCaptionsDataset(split="train")
    val_dataset = ConceptualCaptionsDataset(split="validation")
    
    #dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    #optimizer with different lr
    optimizer = optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": config.text_encoder_lr},
        {"params": [model.temperature, model.bias], "lr": config.head_lr}
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    
    #scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs * len(train_loader)
    )
    
    #mixed precision
    scaler = amp.GradScaler(enabled=config.mixed_precision)
    
    #training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in progress:
            images = batch["image"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            optimizer.zero_grad()
            
            with amp.autocast(enabled=config.mixed_precision):
                logits = model(images, input_ids, attention_mask)
                labels = create_labels(images.size(0), config.device)
                loss = siglip_loss(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        #checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(train_loader),
        }, f"checkpoint_epoch_{epoch+1}.pth")
        
        evaluate(model, val_loader, config)

def evaluate(model, dataloader, config):
    model.eval()
    img2txt_recall = {k: 0 for k in config.k_vals}
    txt2img_recall = {k: 0 for k in config.k_vals}
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            logits = model(images, input_ids, attention_mask)
            similarities = -logits  
            
            batch_size = images.size(0)
            total += batch_size
            
            #img → text retrieval
            for i in range(batch_size):
                scores = similarities[i]
                topk = scores.topk(max(config.k_vals)).indices
                for k in config.k_vals:
                    if i in topk[:k]:
                        img2txt_recall[k] += 1
            
            #text → img retrieval
            for j in range(batch_size):
                scores = similarities[:, j]
                topk = scores.topk(max(config.k_vals)).indices
                for k in config.k_vals:
                    if j in topk[:k]:
                        txt2img_recall[k] += 1
    
    #results
    print("\nEvaluation Results:")
    for k in config.k_vals:
        img2txt = img2txt_recall[k] / total * 100
        txt2img = txt2img_recall[k] / total * 100
        print(f"Image->Text R@{k}: {img2txt:.2f}% | Text->Image R@{k}: {txt2img:.2f}%")
    
    return img2txt_recall, txt2img_recall

if __name__ == "__main__":
    train()