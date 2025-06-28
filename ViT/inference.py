import torch
from PIL import Image
from torchvision import transforms
from models.vit import VisionTransformer
from configs.vit_small import ViT_Small

def classify_image(image_path, model_path, config):
    model = VisionTransformer(config).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(config.device)
    
    #inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probabilities, dim=1).item()
    
    return class_idx, probabilities[0].cpu().numpy()

# EG
if __name__ == "__main__":
    config = ViT_Small()
    config.num_classes = 10  
    class_idx, probs = classify_image(
        "test_image.jpg", 
        "vit_epoch_100.pth", 
        config
    )
    
    print(f"Predicted class: {class_idx}")
    print("Class probabilities:", probs)