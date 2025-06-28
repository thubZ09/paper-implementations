# Vision Transformer (ViT): An Image is Worth 16x16 Words

## Paper Overview

Vision Transformer (ViT) has demonstrated that a pure Transformer applied to sequences of image patches can perform very well on image classification tasks. This work challenged the dominance of convolutional neural networks (CNNs) and opened the door for Transformer-based architectures in vision.

### Key Contributions

- **Pure Transformer Architecture** - First successful application of standard Transformers to image classification without convolutions
- **Patch-based processing** - Treats images as sequences of patches, analogous to words in NLP
- **Scalability** - Shows excellent scaling properties with dataset size and model parameters
- **Transfer learning** - Demonstrates strong transfer learning capabilities from large-scale pre-training
- **Simplicity** - Minimal modifications to the standard Transformer architecture

## Architecture

ViT processes images by dividing them into fixed-size patches and treating each patch as a token in a sequence:

```
Input Image (224×224×3)
         ↓
┌─────────────────────────────────────┐
│        Patch Embedding              │
│  Split into 16×16 patches (196)     │
│  Linear projection to D dimensions  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Add Position Embeddings        │
│  Learnable 1D position embeddings  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│      Add [CLS] Token               │
│  Prepend learnable class token     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Transformer Encoder            │
│  ┌─────────────────────────────┐   │
│  │  Multi-Head Self-Attention  │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │         MLP Block           │   │
│  └─────────────────────────────┘   │
│  × L layers                        │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│    Classification Head             │
│  Extract [CLS] token → Linear → Softmax │
└─────────────────────────────────────┘
```

### Key Components

→ **Patch embedding**: 
   - Split image into non-overlapping patches (typically 16×16)
   - Flatten patches and linearly project to embedding dimension

→ **Position embeddings**: 
   - Learnable 1D position embeddings added to patch embeddings
   - No 2D spatial structure assumed

→ **CLS Token**: 
   - Special learnable token prepended to sequence
   - Used for classification (similar to BERT)

→ **Transformer encoder**: 
   - Standard multi-layer Transformer encoder
   - Multi-head self-attention + MLP blocks
   - Layer normalization and residual connections

→ **Classification head**: 
   - MLP head applied to [CLS] token representation
   - Usually a single linear layer for pre-training

## Training

### Pre-training Strategy

→ **Large-Scale Pre-training**: 
   - Train on large datasets (JFT-300M, ImageNet-21k)
   - Use higher resolution images when possible
   - Standard cross-entropy loss

→ **Fine-tuning**: 
   - Transfer to downstream tasks
   - Often use higher resolution than pre-training
   - Lower learning rates

### Training Config

```python
#for T4 GPU
model_dim = 384  
num_heads = 6
num_layers = 12
mlp_dim = 1536
patch_size = 16
image_size = 224
batch_size = 32 
learning_rate = 1e-3
weight_decay = 0.1
warmup_steps = 1000  
total_steps = 50000  
gradient_accumulation_steps = 4 
```

### Data Augmentation

Pre-training typically uses minimal augmentation:
- **Basic** - Random cropping, horizontal flipping
- **Advanced** - Mixup, CutMix, RandAugment (for better performance)
- **Resolution** - Often pre-train at lower resolution, fine-tune at higher


## Key insights from the paper

→ **Inductive Bias**
- CNNs have strong inductive biases (locality, translation equivariance)
- ViT has minimal inductive bias, relies on data to learn patterns
- Performs poorly on small datasets, excels with large-scale data

→ **Scaling Properties**
- Performance scales well with model size and dataset size
- Larger models benefit more from larger datasets
- No signs of saturation even at very large scales

→ **Transfer Learning**
- Pre-training on large datasets is crucial
- Transfers well to various downstream tasks
- Often outperforms CNNs when properly pre-trained

→ **Computational Efficiency**
- More efficient than ResNets at large scales
- Self-attention complexity is quadratic in sequence length
- Patch size affects computational cost significantly

## Attention visualization

ViT's attention patterns provide insights into what the model focuses on:

```python
def visualize_attention(model, image, layer_idx=11, head_idx=0):
    """visualize attention patterns from a specific layer and head."""
    model.eval()
    with torch.no_grad():
        # get attention weights
        _, attention_weights = model.forward_with_attention(image)
        
        #extract specific layer and head
        attn = attention_weights[layer_idx][0, head_idx]  # [seq_len, seq_len]
        
        #focus on [CLS] token attention to patches
        cls_attention = attn[0, 1:].reshape(14, 14)  # Assuming 224/16 = 14
        
        #visualize
        plt.imshow(cls_attention.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention from [CLS] token - Layer {layer_idx}, Head {head_idx}')
        plt.show()
```

## Variants and Extensions

### Architecture Variants
- **DeiT**: Data-efficient image Transformers with distillation
- **Swin Transformer**: Hierarchical vision Transformer
- **PVT**: Pyramid Vision Transformer
- **CaiT**: Class-Attention in Image Transformers

### Training Improvements
- **DeiT**: Knowledge distillation techniques
- **BEiT**: Masked image modeling pre-training
- **MAE**: Masked autoencoders for self-supervised learning

## 📄 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{dosovitskiy2020image,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
}
```
