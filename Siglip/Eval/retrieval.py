import torch
from torch.utils.data import DataLoader
from training.dataset import ConceptualCaptionsDataset
from models.siglip import SigLIP
from configs.base_config import BaseConfig
from tqdm import tqdm

def evaluate_retrieval(model_path, split="validation"):
    config = BaseConfig()
    model = SigLIP().to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataset = ConceptualCaptionsDataset(split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers
    )
    
    img2txt_recall = {k: 0 for k in config.k_vals}
    txt2img_recall = {k: 0 for k in config.k_vals}
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Retrieval"):
            images = batch["image"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            logits = model(images, input_ids, attention_mask)
            similarities = -logits
            
            batch_size = images.size(0)
            total += batch_size
            
            #img → text retrieval
            for i in range(batch_size):
                scores = similarities[i]
                topk = scores.topk(max(config.k_vals)).indices
                for k in config.k_vals:
                    if i in topk[:k]:
                        img2txt_recall[k] += 1
            
            #text → img retrieval
            for j in range(batch_size):
                scores = similarities[:, j]
                topk = scores.topk(max(config.k_vals)).indices
                for k in config.k_vals:
                    if j in topk[:k]:
                        txt2img_recall[k] += 1
    
    #convert to percentages
    for k in config.k_vals:
        img2txt_recall[k] = img2txt_recall[k] / total * 100
        txt2img_recall[k] = txt2img_recall[k] / total * 100
    
    return img2txt_recall, txt2img_recall