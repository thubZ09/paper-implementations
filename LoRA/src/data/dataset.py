import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional

class IMDBDataset(Dataset):
    """IMDB movie reviews dataset for sentiment analysis fine-tuning"""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        #load IMDB dataset
        dataset = load_dataset("imdb", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        self.data = []
        for item in dataset:
            text = item["text"]
            label = item["label"]  #0: negative, 1: positive
            
            #create input text with sentiment prompt
            sentiment = "positive" if label == 1 else "negative"
            input_text = f"Review: {text}\nSentiment: {sentiment}"
            
            #tokenize
            tokens = self.tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            self.data.append({
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": tokens["input_ids"].squeeze(),  
                "sentiment_label": label
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class AlpacaDataset(Dataset):
    """alternative: Alpaca instruction dataset for instruction following."""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        #load alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        
        #split into train/val (90/10)
        train_size = int(0.9 * len(dataset))
        if split == "train":
            dataset = dataset.select(range(train_size))
        else:
            dataset = dataset.select(range(train_size, len(dataset)))
            
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        self.data = []
        for item in dataset:
            instruction = item["instruction"]
            input_text = item["input"]
            output = item["output"]
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            tokens = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            self.data.append({
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": tokens["input_ids"].squeeze(),
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]