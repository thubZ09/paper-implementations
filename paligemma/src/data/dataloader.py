#dataloader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor
import torch

NUM_WORKERS = os.cpu_count()

class PaliGemmaDataset(Dataset):
    """custom dataset for PaliGemma training with image-text pairs."""
    
    def __init__(self, data_dir: str, processor: PaliGemmaProcessor, split: str = "train"):
        """
        Args:
            data_dir: Path to directory containing image-text pairs
            processor: PaliGemmaProcessor for tokenizing and preprocessing
            split: Dataset split ("train" or "test")
        """
        self.data_dir = data_dir
        self.processor = processor
        self.split = split
        
        # Collect image-text pairs
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load image-text pairs from directory structure."""
        samples = []
        
        # Assuming directory structure: data_dir/class_name/image.jpg
        # with corresponding text files or captions
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # For now, using class name as text prompt
                    # You can modify this to load from text files
                    text = f"What is in this image? {class_name}"
                    target = f"This is a {class_name}."
                    
                    samples.append({
                        'image_path': img_path,
                        'input_text': text,
                        'target_text': target
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Process inputs
        inputs = self.processor(
            text=sample['input_text'],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Process targets
        targets = self.processor(
            text=sample['target_text'],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True
        )
        
        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        targets = {k: v.squeeze(0) for k, v in targets.items()}
        
        return inputs, targets

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    processor: PaliGemmaProcessor,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders for PaliGemma.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        processor: PaliGemmaProcessor for tokenizing and preprocessing.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """

    # Create datasets
    train_dataset = PaliGemmaDataset(train_dir, processor, "train")
    test_dataset = PaliGemmaDataset(test_dir, processor, "test")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    class_names = [d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))]
    
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences."""
        inputs_batch = []
        targets_batch = []
        
        for inputs, targets in batch:
            inputs_batch.append(inputs)
            targets_batch.append(targets)
        
        # Pad sequences to same length within batch
        batch_inputs = processor.tokenizer.pad(
            inputs_batch,
            padding=True,
            return_tensors="pt"
        )
        
        batch_targets = processor.tokenizer.pad(
            targets_batch,
            padding=True,
            return_tensors="pt"
        )
        
        return batch_inputs, batch_targets
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_dataloader, test_dataloader, class_names