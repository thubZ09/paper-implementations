from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T
from configs.base_config import BaseConfig

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, split="train"):
        self.config = BaseConfig()
        self.dataset = load_dataset(
            self.config.dataset_name, 
            split=f"{split}[:{self.config.dataset_subset}%]"
        )
        self.transform = T.Compose([
            T.Resize(self.config.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.text_encoder)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        #load 
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)
        
        #tokenize text
        text = item["caption"]
        tokens = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }