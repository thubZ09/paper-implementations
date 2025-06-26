import torch

def clip_loss(logits, device):
    labels = torch.arange(logits.shape[0], device=device)
    loss_i = torch.nn.functional.cross_entropy(logits, labels)
    loss_t = torch.nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask)
        loss = clip_loss(logits, device)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(images, input_ids, attention_mask)
            loss = clip_loss(logits, device)
            total_loss += loss.item()
    
    return total_loss / len(loader)