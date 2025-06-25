import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import RobertaTokenizer

class CLIPDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, max_length=77):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        caption = row['caption']
        image_name = row['image']
        
        #image loading
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        #apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        #tokenize 
        text = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': text['input_ids'].squeeze(0),
            'attention_mask': text['attention_mask'].squeeze(0)
        }