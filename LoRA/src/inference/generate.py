import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Optional, Dict, Any
import numpy as np

class LoRAInference:
    """inference class for LoRA fine-tuned models"""
    
    def __init__(self, model, tokenizer_name: str, device: str = "cuda"):
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.model.to(device)
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
    ) -> List[str]:
        """generate text using the fine-tuned model."""
        
        #tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        #set pad token
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        #decode generated text
        generated_texts = []
        for output in outputs:
            #remove input tokens from output
            generated_ids = output[inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """perform sentiment analysis on text (for IMDB-trained models)"""
        
        #format input for sentiment analysis
        prompt = f"Review: {text}\nSentiment:"
        
        #tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            last_token_logits = logits[0, -1, :]
            
            #get token IDs for "+" "-"
            positive_id = self.tokenizer.encode(" positive", add_special_tokens=False)[0]
            negative_id = self.tokenizer.encode(" negative", add_special_tokens=False)[0]
            
            #extract logits for sentiment tokens
            sentiment_logits = torch.stack([
                last_token_logits[negative_id],  
                last_token_logits[positive_id] 
            ])
            
            #apply softmax
            sentiment_probs = F.softmax(sentiment_logits, dim=0)
            
            return {
                "negative": sentiment_probs[0].item(),
                "positive": sentiment_probs[1].item(),
                "predicted_sentiment": "positive" if sentiment_probs[1] > sentiment_probs[0] else "negative"
            }
    
    def instruction_following(self, instruction: str, input_text: str = "") -> str:
        """generate response for instruction-following tasks (for Alpaca-trained models)"""
        
        # Format prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        #generate response
        responses = self.generate_text(
            prompt,
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        return responses[0] if responses else ""
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 200,
        temperature: float = 0.7,
        batch_size: int = 4
    ) -> List[str]:
        """generate text for multiple prompts in batches"""
        
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            #tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            #decode results
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].shape[0]
                generated_ids = output[input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_results.append(generated_text.strip())
        
        return all_results
    
    def interactive_chat(self):
        """interactive chat interface."""
        print("LoRA Chat Interface (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            #generate response
            try:
                responses = self.generate_text(
                    user_input,
                    max_length=150,
                    temperature=0.8,
                    top_p=0.9
                )
                
                print(f"Bot: {responses[0]}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error generating response: {e}")
                continue