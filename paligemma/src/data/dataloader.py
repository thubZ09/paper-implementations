import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor
import torch

NUM_WORKERS = os.cpu_count()

def read_flickr8k_captions(captions_file):
    """reads captions.txt and returns a dict: image → [captions]"""
    from collections import defaultdict
    mapping = defaultdict(list)
    with open(captions_file, "r") as f:
        for line in f:
            img, caption = line.strip().split('\t')
            img = img.split('#')[0]
            mapping[img].append(caption)
    return dict(mapping)

def read_split_list(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]

class Flickr8kDataset(Dataset):

    def __init__(self, images_dir, captions_dict, split_list, processor, max_length=128):

        self.images_dir = images_dir
        self.captions_dict = captions_dict
        self.split_list = split_list
        self.processor = processor
        self.max_length = max_length

        self.samples = []
        for img_fname in split_list:
            if img_fname in captions_dict:
                for caption in captions_dict[img_fname]:
                    self.samples.append((img_fname, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_fname, caption = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_fname)
        image = Image.open(img_path).convert('RGB')

#prompt
        input_text = "Describe the image."
        target_text = caption

        inputs = self.processor(
            text=input_text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        targets = self.processor(
            text=target_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        targets = {k: v.squeeze(0) for k, v in targets.items()}
        return inputs, targets

def collate_fn(batch, processor):
    """Pads a batch of examples for dataLoader"""
    inputs_batch = [inputs for inputs, targets in batch]
    targets_batch = [targets for inputs, targets in batch]
    batch_inputs = processor.tokenizer.pad(
        inputs_batch, padding=True, return_tensors="pt"
    )
    batch_targets = processor.tokenizer.pad(
        targets_batch, padding=True, return_tensors="pt"
    )
    return batch_inputs, batch_targets

def create_dataloaders(
    config,
    processor,
):
    """creates train/val DataLoaders for Flickr8k dataset using config dict."""

    image_dir = config['data']['image_dir']
    captions_file = config['data']['captions_file']
    train_list = config['data']['train_list']
    val_list = config['data']['val_list']
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('dataloader_num_workers', 2)
    max_length = config['data'].get('max_length', 128)

    captions_dict = read_flickr8k_captions(captions_file)
    train_split = read_split_list(train_list)
    val_split = read_split_list(val_list)

    train_dataset = Flickr8kDataset(
        image_dir, captions_dict, train_split, processor, max_length=max_length
    )
    val_dataset = Flickr8kDataset(
        image_dir, captions_dict, val_split, processor, max_length=max_length
    )

    # Collate with processor
    collate = lambda batch: collate_fn(batch, processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate
    )

    class_names = ["caption"] 

    return train_loader, val_loader, class_names