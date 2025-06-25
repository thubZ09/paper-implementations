import torch
from torch.utils.data import DataLoader, random_split
from dataset import CLIPDataset
from model import CLIP
from engine import train_epoch, validate
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os

#config
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
projection_dim = 256
lr = 5e-5
epochs = 30

def prepare_datasets():
    captions = pd.read_csv('captions.csv')  
    
    full_dataset = CLIPDataset(
        dataframe=captions,
        image_dir='images',
        transform=A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        max_length=77
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

#dataloaders
train_dataset, val_dataset = prepare_datasets()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = CLIP(projection_dim=projection_dim).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=0.01
)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print("-" * 50)

torch.save(model.state_dict(), "clip_model.pth")