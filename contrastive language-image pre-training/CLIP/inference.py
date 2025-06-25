import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import RobertaTokenizer
from model import CLIP  # Assuming model.py contains CLIP class
import argparse

#config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

class CLIPInference:
    def __init__(self, model_path, projection_dim=256):
        self.model = CLIP(projection_dim=projection_dim).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """load and preprocess image"""
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(DEVICE)
    
    def preprocess_text(self, texts):
        """tokenizelist of text prompts"""
        return self.tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt",
            max_length=77,
            truncation=True
        )
    
    def classify_image(self, image, classes, prompt_template="a photo of a {}"):

        if isinstance(image, str):
            image = self.preprocess_image(image)

        text_prompts = [prompt_template.format(c) for c in classes]
        text_inputs = self.preprocess_text(text_prompts)
        
        #move inputs to device
        input_ids = text_inputs["input_ids"].to(DEVICE)
        attention_mask = text_inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            image_features = self.model.image_encoder(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            text_features = self.model.text_encoder(input_ids, attention_mask)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            logits = (image_features @ text_features.t()) * logit_scale

        probs = logits.softmax(dim=-1).cpu().numpy().flatten()

        sorted_indices = np.argsort(probs)[::-1]
        results = {
            "predictions": [
                {"class": classes[i], "probability": float(probs[i])} 
                for i in sorted_indices
            ],
            "top_prediction": classes[sorted_indices[0]],
            "top_probability": float(probs[sorted_indices[0]])
        }
        
        return results
    
    def image_text_similarity(self, image_path, text_prompt):

        image = self.preprocess_image(image_path)

        text_inputs = self.preprocess_text([text_prompt])

        input_ids = text_inputs["input_ids"].to(DEVICE)
        attention_mask = text_inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():

            image_features = self.model.image_encoder(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            text_features = self.model.text_encoder(input_ids, attention_mask)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.t()).item()
        
        return similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lip zero-shot inference")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="path to trained CLIP")
    parser.add_argument("--image", type=str, required=True, 
                        help="path to input image")
    parser.add_argument("--classes", type=str, 
                        default="dog,cat,bird,car,tree,person",
                        help="comma-separated list of classes")
    parser.add_argument("--projection-dim", type=int, default=256,
                        help="projection dimension used in CLIP")
    
    args = parser.parse_args()

    clip_inference = CLIPInference(
        model_path=args.model_path,
        projection_dim=args.projection_dim
    )

    classes = [c.strip() for c in args.classes.split(",")]

    results = clip_inference.classify_image(args.image, classes)
    
    print("\nZero-Shot Classification Results:")
    print(f"Image: {args.image}")
    print(f"Top Prediction: {results['top_prediction']} ({results['top_probability']*100:.2f}%)")
    
    print("\nAll Predictions:")
    for pred in results["predictions"]:
        print(f"- {pred['class']}: {pred['probability']*100:.2f}%")