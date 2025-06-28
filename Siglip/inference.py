import torch
from PIL import Image
from torchvision import transforms
from models.siglip import SigLIP
from configs.base_config import BaseConfig

def image_to_text(image_path, candidate_texts, model_path):
    config = BaseConfig()
    model = SigLIP().to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(config.device)
    
    #get embeddings
    with torch.no_grad():
        image_embed = model.encode_image(image_tensor)
        text_embeds = model.encode_text(candidate_texts)
    
    # calculate similarities
    similarities = F.cosine_similarity(image_embed, text_embeds)
    probs = torch.softmax(similarities * 100, dim=-1)  
    
    #resul
    results = []
    for i, text in enumerate(candidate_texts):
        results.append({
            "text": text,
            "similarity": similarities[i].item(),
            "probability": probs[i].item()
        })
    
    #rt by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results

# E.g
if __name__ == "__main__":
    model_path = "best_model.pth"
    image_path = "test_image.jpg"
    candidates = [
        "a cat sleeping on a sofa",
        "a dog playing in the park",
        "a sunset over the ocean",
        "a city skyline at night"
    ]
    
    results = image_to_text(image_path, candidates, model_path)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['text']} (Similarity: {res['similarity']:.4f}, Prob: {res['probability']:.2%})")