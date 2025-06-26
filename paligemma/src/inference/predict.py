#predict.py
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PaliGemmaProcessor
from typing import List, Dict, Optional
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_and_plot_image(
    model: torch.nn.Module,
    processor: PaliGemmaProcessor,
    image_path: str,
    prompt: str = "What is in this image?",
    max_new_tokens: int = 50,
    device: torch.device = device,
    class_names: Optional[List[str]] = None
):
    """Predicts on a target image with a PaliGemma model.

    Args:
        model: A trained PaliGemma model.
        processor: PaliGemmaProcessor for preprocessing.
        image_path: Filepath to target image to predict on.
        prompt: Text prompt to use for generation.
        max_new_tokens: Maximum number of tokens to generate.
        device: Target device to perform prediction on.
        class_names: Optional list of class names for classification tasks.
    """
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make sure model is on target device and in eval mode
    model.to(device)
    model.eval()
    
    # Generate prediction
    with torch.inference_mode():
        if hasattr(model, 'generate'):
            # For generation tasks
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            
            # Decode the generated text
            generated_text = processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )[0]
            
            # Plot results
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.title(f"Prompt: {prompt}\nGenerated: {generated_text}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return generated_text
            
        else:
            # For classification tasks
            logits = model(inputs).logits
            
            if class_names and logits.shape[-1] == len(class_names):
                # Classification mode
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1)
                pred_class = class_names[pred_idx.item()]
                confidence = probs.max().item()
                
                # Plot results
                plt.figure(figsize=(10, 6))
                plt.imshow(image)
                plt.title(f"Predicted: {pred_class} (Confidence: {confidence:.3f})")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
                return pred_class, confidence
            else:
                # Text generation from logits
                predicted_ids = torch.argmax(logits, dim=-1)
                generated_text = processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                plt.figure(figsize=(10, 6))
                plt.imshow(image)
                plt.title(f"Prompt: {prompt}\nGenerated: {generated_text}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
                return generated_text

def batch_predict(
    model: torch.nn.Module,
    processor: PaliGemmaProcessor,
    image_paths: List[str],
    prompts: List[str],
    max_new_tokens: int = 50,
    device: torch.device = device
) -> List[str]:
    """Make batch predictions on multiple images.
    
    Args:
        model: A trained PaliGemma model.
        processor: PaliGemmaProcessor for preprocessing.
        image_paths: List of image file paths.
        prompts: List of text prompts (same length as image_paths).
        max_new_tokens: Maximum number of tokens to generate.
        device: Target device to perform prediction on.
        
    Returns:
        List of generated text responses.
    """
    
    if len(image_paths) != len(prompts):
        raise ValueError("Number of images and prompts must match")
    
    model.to(device)
    model.eval()
    
    results = []
    
    with torch.inference_mode():
        for img_path, prompt in zip(image_paths, prompts):
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Process inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            if hasattr(model, 'generate'):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
                generated_text = processor.batch_decode(
                    generated_ids[:, inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )[0]
            else:
                logits = model(inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                generated_text = processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
            
            results.append(generated_text)
    
    return results

def evaluate_on_dataset(
    model: torch.nn.Module,
    processor: PaliGemmaProcessor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = device,
    max_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model performance on a dataset.
    
    Args:
        model: A trained PaliGemma model.
        processor: PaliGemmaProcessor for preprocessing.
        dataloader: DataLoader for evaluation dataset.
        device: Target device.
        max_samples: Maximum number of samples to evaluate.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    
    model.to(device)
    model.eval()
    
    total_samples = 0
    total_loss = 0.0
    correct_predictions = 0
    
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if total_samples >= max_samples:
                break
                
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            logits = outputs.logits
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = targets['input_ids'].view(-1)
            
            # Mask padding tokens
            mask = labels_flat != -100
            if mask.any():
                logits_masked = logits_flat[mask]
                labels_masked = labels_flat[mask]
                
                loss = torch.nn.functional.cross_entropy(logits_masked, labels_masked)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits_masked, dim=-1)
                correct_predictions += (predictions == labels_masked).sum().item()
                total_samples += labels_masked.numel()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total_samples
    }