import os
import sys
import torch
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ModelConfig
from src.models.lora_model import LoRAGPT2
from src.data.dataset import IMDBDataset, AlpacaDataset
from src.inference.generate import LoRAInference
from src.training.utils import log_gpu_memory, save_metrics

def evaluate_sentiment_model(model, dataset, tokenizer_name, device="cuda"):
    """evaluate sentiment analysis model"""
    inference = LoRAInference(model, tokenizer_name, device)
    
    predictions = []
    true_labels = []
    
    print("evaluating sentiment analysis...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        #get the original text (we need to extract it from the formatted input)
        input_ids = item["input_ids"]
        text = dataset.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        #extract the review text (before "sentiment")
        if "Review:" in text and "Sentiment:" in text:
            review_text = text.split("Review:")[1].split("Sentiment:")[0].strip()
        else:
            review_text = text
        
        #fet prediction
        try:
            sentiment_result = inference.sentiment_analysis(review_text)
            predicted_label = 1 if sentiment_result["predicted_sentiment"] == "positive" else 0
            predictions.append(predicted_label)
            true_labels.append(item["sentiment_label"])
        except Exception as e:
            print(f"error processing item {i}: {e}")
            continue
    
    #calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)
    
    metrics = {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics

def evaluate_language_model(model, dataset, tokenizer_name, device="cuda"):
    """evaluate language model using perplexity"""
    model.eval()
    model.to(device)
    
    total_loss = 0
    total_tokens = 0
    
    print("evaluating language model....")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            labels = item["labels"].unsqueeze(0).to(device)
            
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                #non-padding tokens
                non_pad_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    metrics = {
        "average_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens
    }
    
    return metrics

def evaluate_generation_quality(model, tokenizer_name, prompts, device="cuda"):
    """evaluate generation quality with sample prompts"""
    inference = LoRAInference(model, tokenizer_name, device)
    
    results = []
    
    print("evaluating..")
    for prompt in tqdm(prompts):
        try:
            generated_texts = inference.generate_text(
                prompt,
                max_length=100,
                temperature=0.7,
                num_return_sequences=3
            )
            
            results.append({
                "prompt": prompt,
                "generated_texts": generated_texts
            })
            
        except Exception as e:
            print(f"error generating for prompt '{prompt}': {e}")
            continue
    
    return results

def plot_confusion_matrix(cm, labels, output_dir):
    """plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('confusion Matrix')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"confusion matrix saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the fine-tuned model")
    parser.add_argument("--dataset", type=str, default="imdb", 
                       choices=["imdb", "alpaca"], help="Dataset to evaluate on")
    parser.add_argument("--eval_samples", type=int, default=1000, 
                       help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation", 
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    #output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = ModelConfig()
    print("loading model...")
    model = LoRAGPT2(config)
    
    #load LoRA weights
    if os.path.exists(args.model_path):
        if os.path.isdir(args.model_path):
            lora_weights_path = os.path.join(args.model_path, "lora_weights.pt")
        else:
            lora_weights_path = args.model_path
        
        model.load_lora_weights(lora_weights_path)
        print(f"loaded LoRA weights from {lora_weights_path}")
    else:
        print(f"model path {args.model_path} does not exist!")
        return
    
    #load eval dataset
    print("loading evaluation dataset...")
    if args.dataset == "imdb":
        eval_dataset = IMDBDataset(
            split="test",
            tokenizer_name=config.model_name,
            max_length=config.max_length,
            max_samples=args.eval_samples
        )
    else:
        eval_dataset = AlpacaDataset(
            split="val",
            tokenizer_name=config.model_name,
            max_length=config.max_length,
            max_samples=args.eval_samples
        )
    
    print(f"Loaded {len(eval_dataset)} samples for evaluation")
    
    log_gpu_memory()
    
    #evaluate model
    all_metrics = {}
    
    if args.dataset == "imdb":
        #sentiment analysis evaluation
        sentiment_metrics = evaluate_sentiment_model(
            model, eval_dataset, config.model_name, args.device
        )
        all_metrics["sentiment_analysis"] = sentiment_metrics
        
        #plot confusion matrix
        cm = np.array(sentiment_metrics["confusion_matrix"])
        plot_confusion_matrix(cm, ["Negative", "Positive"], args.output_dir)
        
        print(f"Sentiment Analysis Results:")
        print(f"Accuracy: {sentiment_metrics['accuracy']:.4f}")
        print(f"Precision: {sentiment_metrics['precision']:.4f}")
        print(f"Recall: {sentiment_metrics['recall']:.4f}")
        print(f"F1-Score: {sentiment_metrics['f1_score']:.4f}")
    
    #language model evaluation
    lm_metrics = evaluate_language_model(
        model, eval_dataset, config.model_name, args.device
    )
    all_metrics["language_model"] = lm_metrics
    
    print(f"Language Model Results:")
    print(f"Perplexity: {lm_metrics['perplexity']:.2f}")
    print(f"Average Loss: {lm_metrics['average_loss']:.4f}")
    
    #generation quality evaluation
    if args.dataset == "imdb":
        test_prompts = [
            "This movie was absolutely",
            "I think this film is",
            "The acting in this movie",
            "Overall, I would say this movie is",
            "The plot of this movie"
        ]
    else:
        test_prompts = [
            "### Instruction:\nWrite a short story about a robot.\n\n### Response:\n",
            "### Instruction:\nExplain how photosynthesis works.\n\n### Response:\n",
            "### Instruction:\nWhat are the benefits of exercise?\n\n### Response:\n"
        ]
    
    generation_results = evaluate_generation_quality(
        model, config.model_name, test_prompts, args.device
    )
    all_metrics["generation_samples"] = generation_results
    
    print(f"Generation Examples:")
    for result in generation_results[:2]:  # Show first 2 examples
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_texts'][0]}")
        print("-" * 50)
    
    #save metrics nd summary
    save_metrics(all_metrics, args.output_dir, "evaluation_results.json")
    summary = {
        "dataset": args.dataset,
        "eval_samples": len(eval_dataset),
        "model_path": args.model_path,
        "config": config.__dict__
    }
    
    if args.dataset == "imdb":
        summary["accuracy"] = sentiment_metrics["accuracy"]
        summary["f1_score"] = sentiment_metrics["f1_score"]
    
    summary["perplexity"] = lm_metrics["perplexity"]
    summary["average_loss"] = lm_metrics["average_loss"]
    
    save_metrics(summary, args.output_dir, "evaluation_summary.json")
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()