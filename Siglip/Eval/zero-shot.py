import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from models.siglip import SigLIP
from configs.base_config import BaseConfig

def zero_shot_eval(model_path, dataset_path):
    config = BaseConfig()
    model = SigLIP().to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    #load imageNet validation set
    dataset = ImageNet(root=dataset_path, split="val")
    
    class_prompts = [f"a photo of a {class_name}" for class_name in dataset.classes]
    text_embeddings = model.encode_text(class_prompts)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    #create dataloader
    transform = T.Compose([
        T.Resize(config.image_size),
        T.CenterCrop(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Zero-shot Evaluation"):
            images = images.to(config.device)
            image_embeddings = model.encode_image(images)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            
            #calculate similarities
            similarities = image_embeddings @ text_embeddings.T
            predictions = similarities.argmax(dim=-1)
            
            correct += (predictions.cpu() == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    print(f"Zero-shot Accuracy: {accuracy:.2f}%")
    return accuracy