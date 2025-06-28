import torch
import torch.nn as nn
from tqdm import tqdm
from training.dataset import get_dataset

def evaluate(model_path, config, dataset_name='cifar10'):
    #model
    model = VisionTransformer(config).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    #dataset
    val_dataset, _ = get_dataset(dataset_name, config.image_size, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    #validation
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, config)
    print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
    return val_acc

def validate(model, val_loader, criterion, cfg):
    pass