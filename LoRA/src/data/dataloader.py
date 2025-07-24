import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Any, Callable
import random
from transformers import AutoTokenizer

class DataCollatorForLanguageModeling:
    """data collator for language modeling tasks."""
     
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        #handle dict or list of examples
        if isinstance(examples[0], dict):
            batch = self._tensorize_batch(examples)
        else:
            batch = {"input_ids": examples}
        
        #pad sequences
        batch = self._pad_sequences(batch)
        
        #creating labels for language modeling
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        
        #apply MLM if specified
        if self.mlm:
            batch["input_ids"], batch["labels"] = self._mask_tokens(
                batch["input_ids"], batch["labels"]
            )
        
        return batch
    
    def _tensorize_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """convert examples to tensors"""
        batch = {}
        
        #get all keys from the first example
        keys = examples[0].keys()
        
        for key in keys:
            values = [example[key] for example in examples]
            
            if key in ["input_ids", "attention_mask", "labels"]:
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        
        return batch
    
    def _pad_sequences(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """pad sequences to the same length"""
        max_length = max(len(seq) for seq in batch["input_ids"])
        
        #pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        #pad input_ids
        padded_input_ids = []
        attention_masks = []
        
        for seq in batch["input_ids"]:
            pad_length = max_length - len(seq)
            if pad_length > 0:
                padded_seq = torch.cat([
                    seq,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=seq.dtype)
                ])
                attention_mask = torch.cat([
                    torch.ones(len(seq), dtype=torch.long),
                    torch.zeros(pad_length, dtype=torch.long)
                ])
            else:
                padded_seq = seq
                attention_mask = torch.ones(len(seq), dtype=torch.long)
            
            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)
        
        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(attention_masks)
        
        #pad labels if present
        if "labels" in batch:
            padded_labels = []
            for seq in batch["labels"]:
                pad_length = max_length - len(seq)
                if pad_length > 0:
                    padded_seq = torch.cat([
                        seq,
                        torch.full((pad_length,), -100, dtype=seq.dtype)  # -100 is ignored in loss
                    ])
                else:
                    padded_seq = seq
                padded_labels.append(padded_seq)
            
            batch["labels"] = torch.stack(padded_labels)
        
        return batch
    
    def _mask_tokens(
        self, 
        inputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """apply masking for MLM."""
        if self.tokenizer.mask_token is None:
            raise ValueError("tokenizer does not have a mask token for MLM")
        
        #create random mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        #don't mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_token_id in self.tokenizer.all_special_ids:
            special_tokens_mask |= (labels == special_token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        #only predict masked tokens
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
                
        return inputs, labels

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """create a DataLoader with common settings"""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs
    )

def create_balanced_dataloader(
    dataset: Dataset,
    batch_size: int,
    class_weights: Optional[Dict[int, float]] = None,
    **kwargs
) -> DataLoader:
    """create a balanced dataLoader using weighted sampling"""
    
    #get labels from dataset
    if hasattr(dataset, 'get_labels'):
        labels = dataset.get_labels()
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, dict) and 'sentiment_label' in item:
                labels.append(item['sentiment_label'])
            else:
                raise ValueError("Cannot extract labels from dataset")
    
    #calculate class weights if not provided
    if class_weights is None:
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = {i: total_samples / count for i, count in enumerate(class_counts)}
    
    #create sample weights
    sample_weights = [class_weights[label] for label in labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **kwargs
    )

class InfiniteDataLoader:
    """dataLoader that cycles infinitely through the dataset."""
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

def get_dataloader_stats(dataloader: DataLoader) -> Dict[str, Any]:
    """get statistics about a dataLoader."""
    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    
    #sample a batch to get tensor info
    sample_batch = next(iter(dataloader))
    
    stats = {
        "total_batches": total_batches,
        "total_samples": total_samples,
        "batch_size": dataloader.batch_size,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last
    }
    
    #add tensor shapes
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            stats[f"{key}_shape"] = list(value.shape)
            stats[f"{key}_dtype"] = str(value.dtype)
    
    return stats