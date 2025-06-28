import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

def visualize_attention(model, image_path, config, layer=11, head=0):
    #load image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(config.device)
    
    model.eval()
    with torch.no_grad():
        x = model.patch_embed(img_tensor)

        attention_weights = []
        def hook(module, input, output):
            attention_weights.append(output[1])
        
        handle = model.blocks[layer].attention.fn.fn.register_forward_hook(hook)
        
        #forward pass
        for i, block in enumerate(model.blocks):
            if i == layer:
                x = block(x)
            else:
                with torch.no_grad():
                    x = block(x)

        handle.remove()

    attn = attention_weights[0]
    attn = attn[head]  
    cls_attn = attn[0, 1:]  
    size = int(np.sqrt(cls_attn.size(0)))
    cls_attn = cls_attn.reshape(size, size).cpu().numpy()
    
    #visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cls_attn, cmap='viridis')
    plt.colorbar()
    plt.title(f"Attention Map - Layer {layer}, Head {head}")
    plt.savefig("attention_map.png")
    plt.close()
    
    return cls_attn