# CLIP Implementation

PyTorch implementation of the CLIP (Contrastive Language-Image Pretraining) from the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).


## Overview

CLIP is a often reffered to as a foundation of today's multimodal models that learns visual concepts from natural language supervision. Unlike traditional computer vision models that require labeled datasets, it learns by associating images with their natural language descriptions. This enables:

1. **Zero-shot transfer** to new visual tasks
2. **Natural language interface** for image classification
3. **Strong generalization** across diverse datasets

## Key Innovations from the paper 
### Contrastive Pre-training
CLIP is trained using a contrastive objective that predicts which caption goes with which image in a batch. This creates a shared embedding space where:

- similar images and texts are close
- dissimilar pairs are far apart

### Web-Scale Training
The original CLIP was trained on:
- 400 million image-text pairs
- Collected from the internet
- Covering diverse concepts

### Efficient Architecture
- **Image Encoder**: Vision Transformer (ViT)
- **Text Encoder**: Transformer (similar to GPT)
- **Projection Layers**: Map both modalities to shared space

### Image Encoder
- Based on Vision Transformer (ViT)
- Input: 224x224 images
- Dvided into 16x16 patches
- Output: [CLS] token embedding

### Text Encoder
- Based on RoBERTa architecture
- Input: 77 token sequences
- Output: [EOS] token embedding

### Projection
Linear layers map both encoders' outputs to:
- shared embedding space
- fixed dimension (256 in our implementation)

### Training Process
#### Loss Function
- Symmetric contrastive loss:

```python
def clip_loss(logits):
    labels = torch.arange(batch_size)
    loss_i = cross_entropy(logits, labels)  # Image->Text
    loss_t = cross_entropy(logits.t(), labels)  # Text->Image
    return (loss_i + loss_t)/2
```
#### Optimization
- AdamW optimizer
- cosine learning rate schedule
- Large batch sizes (up to 32k in original)

#### Evaluation
| Dataset   | Top-1 Accuracy |
|-----------|:--------------:|
| ImageNet  | 76.2%          |
| CIFAR-10  | 94.9%          |
| CIFAR-100 | 77.0%          |
| STL-10    | 98.7%          |
| Food-101  | 92.7%          |

## Key differences from original
- Smaller projection dimension (256 vs 512)
- Fewer training examples
- Simplified transformer architecture
- Faster training on single GPU

## References
- Radford, A., et al. (2021). **Learning Transferable Visual Models From Natural Language Supervision**. arXiv:2103.00020

- Dosovitskiy, A., et al. (2020). **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**. arXiv:2010.11929