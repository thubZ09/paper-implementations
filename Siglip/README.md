# SigLIP: Sigmoid Loss for Language Image Pre-Training

## Paper Overview

SigLIP introduces a simple/effective alternative to the contrastive loss used in CLIP. Instead of using softmax-based contrastive learning with large batch sizes, SigLIP employs a sigmoid-based loss function that operates on image-text pairs independently. This approach eliminates the need for large batch sizes while maintaining competitive performance.

### Key Contributions

- **Sigmoid Loss Function** - Replaces the InfoNCE loss with a simple sigmoid loss that doesn't require negative sampling across the batch
- **Improved Efficiency** - Eliminates the quadratic memory scaling with batch size, enabling training with smaller batches
- **Better Performance** - Achieves superior results compared to CLIP on various benchmarks despite the simpler loss function
- **Scalability** - More practical for scenarios with limited computational resources

## Architecture

SigLIP follows the same dual-encoder architecture as CLIP:

```
┌─────────────────┐    ┌─────────────────┐
│   Text Encoder  │    │  Image Encoder  │
│                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Tokenizer │ │    │ │  Vision     │ │
│ │             │ │    │ │  Transformer│ │
│ │   BERT-like │ │    │ │  (ViT)      │ │
│ │ Transformer │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │
│  Text Features  │    │ Image Features  │
│     [d_model]   │    │    [d_model]    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌─────────────┐
              │  Sigmoid    │
              │  Loss       │
              │  Function   │
              └─────────────┘
```

### Components

- **Image Encoder** - Vision Transformer (ViT) that processes images into feature representations
- **Text Encoder** - Transformer-based encoder (similar to BERT) for text processing
- **Projection Heads** - Linear layers that map encoded features to a shared embedding space
- **Sigmoid Loss** - A pairwise sigmoid loss instead of contrastive loss

## Loss Function

The core innovation of SigLIP is the sigmoid loss function:

```
L_sigmoid = -log(σ(z_i * y_i))
```

Where:
- `z_i` is the cosine similarity between image and text embeddings
- `y_i` is the label (+1 for positive pairs, -1 for negative pairs)
- `σ` is the sigmoid function

### Comparison with CLIP's InfoNCE Loss

| Aspect | CLIP (InfoNCE) | SigLIP (Sigmoid) |
|--------|----------------|------------------|
| **Batch Dependency** | Requires large batches | Independent pairs |
| **Memory Scaling** | O(B²) | O(B) |
| **Negative Sampling** | In-batch negatives | Global negatives |
| **Temperature** | Learnable parameter | Not required |
| **Implementation** | Complex | Simple |

## Training

### Procedure

1. **Data Preparation** - Image-text pairs from large-scale datasets
2. **Batch Construction** - Unlike CLIP, batch size can be much smaller
3. **Forward Pass** - 
   - Encode images and texts separately
   - Compute cosine similarity between all pairs
   - Apply sigmoid loss with positive/negative labels
4. **Optimization** - Standard gradient descent with Adam optimizer

### Training Configs

```python
# Typical training hyperparameters
batch_size = 256  # much smaller than CLIP
learning_rate = 1e-4
warmup_steps = 2000
total_steps = 100000
weight_decay = 0.1
```

### Data Augmentation

- **Images** - Random cropping, horizontal flipping, color jittering
- **Text** - No augmentation typically applied
- **Negative Sampling** - Global negative sampling strategy

## Eval

### Benchmark results

SigLIP demonstrates strong performance across multiple benchmarks:

| Dataset | CLIP (ViT-B/32) | SigLIP (ViT-B/32) | Improvement |
|---------|-----------------|-------------------|-------------|
| ImageNet | 63.2% | 64.6% | +1.4% |
| CIFAR-10 | 95.6% | 96.2% | +0.6% |
| CIFAR-100 | 77.8% | 79.1% | +1.3% |
| Flickr30k | 85.4% | 86.7% | +1.3% |

### Zero-Shot Classification

```python
# Example zero-shot evaluation
model.eval()
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(class_templates)
    
    similarities = image_features @ text_features.T
    predictions = similarities.argmax(dim=-1)
```

### Image-Text Retrieval

- **Image-to-Text Retrieval** - Find relevant captions for given images
- **Text-to-Image Retrieval** - Find relevant images for given text queries
- **Metrics** - Recall@1, Recall@5, Recall@10

## Usage

### from scratch

```bash
python training/train.py \
    --config configs/base_config.py \
    --data-path /path/to/dataset \
    --batch-size 256 \
    --learning-rate 1e-4 \
    --epochs 50 \
    --output-dir ./checkpoints
```

### fine-tune

```bash
python training/train.py \
    --config configs/base_config.py \
    --pretrained-path ./checkpoints/siglip_base.pt \
    --data-path /path/to/dataset \
    --batch-size 128 \
    --learning-rate 5e-5 \
    --epochs 10
```

### Eval

```bash
# Zero-shot classification
python evaluation/zero_shot.py \
    --model-path ./checkpoints/siglip_base.pt \
    --dataset imagenet \
    --data-path /path/to/imagenet

# Image-text retrieval
python evaluation/retrieval.py \
    --model-path ./checkpoints/siglip_base.pt \
    --dataset flickr30k \
    --data-path /path/to/flickr30k
```

## Key differences from CLIP

1. **Loss function** - Sigmoid loss vs. InfoNCE contrastive loss
2. **Batch size requirements** - Can work with much smaller batches
3. **Memory Efficiency** - Linear scaling instead of quadratic
4. **Negative sampling** - More flexible negative sampling strategies
5. **Training stability** - More stable training with smaller batches

## Performance Tips

- **Batch size** - Start with smaller batches (256-512) and scale up if needed
- **Learning rate** - Use warmup and cosine annealing for better convergence
- **Data quality** - High-quality image-text pairs are crucial
- **Negative sampling** - Experiment with different negative sampling ratios
- **Regularization** - Apply appropriate weight decay and dropout

## Citation

```bibtex
@article{zhai2023sigmoid,
  title={Sigmoid Loss for Language Image Pre-Training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  journal={arXiv preprint arXiv:2303.15343},
  year={2023}
}
```
    
