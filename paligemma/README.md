# PaliGemma: A 3B Vision-Language Model 

## 📄 Paper Overview

- **Authors**: Google Research Team
- **Paper**: [arXiv:2407.07726](https://arxiv.org/abs/2407.07726)
- **Release**: July 2024

PaliGemma is a 3-billion parameter vision-language model that combines a SigLIP vision encoder with a Gemma language model. It's designed for a variety of vision-language tasks including image captioning, visual question answering, object detection, and optical character recognition.

## 📌 Architecture

PaliGemma follows a standard vision-language model architecture with three main components:

### 1. Vision Encoder: SigLIP-So400m/14
- **Base Model**: SigLIP (Sigmoid Loss for Language Image Pre-training)
- **Parameters**: ~400M parameters
- **Input Resolution**: 224×224 (14×14 patches)
- **Output**: 256 visual tokens of dimension 1152
- **Key Features**:
  - Uses sigmoid loss instead of softmax for better efficiency
  - Trained on web-scale image-text pairs
  - Frozen during PaliGemma training

### 2. Vision-Language Connector
- **Purpose**: Projects vision tokens to language model dimension
- **Architecture**: Linear projection layer
- **Input**: 256 × 1152 (vision tokens)
- **Output**: 256 × 2048 (language model dimension)
- **Trainable**: Yes

### 3. Language Model: Gemma-2B
- **Base Model**: Gemma-2B decoder-only transformer
- **Parameters**: ~2B parameters
- **Context Length**: 8192 tokens
- **Architecture**:
  - 18 transformer layers
  - 16 attention heads
  - Hidden dimension: 2048
  - Vocabulary size: 256,000 tokens
- **Modifications**: 
  - Prefix LM attention pattern for vision tokens
  - Standard causal attention for text tokens

### 👇Overall Architecture Flow
```
Input Image (224×224) 
    ↓
SigLIP Vision Encoder 
    ↓
Visual Tokens (256 × 1152)
    ↓
Linear Projection
    ↓
Projected Tokens (256 × 2048)
    ↓
[Concatenate with Text Tokens]
    ↓
Gemma Language Model
    ↓
Output Text
```

## 🔧 Training Process

### Stage 1: Unimodal Pre-training
- **Vision Encoder**: Pre-trained SigLIP frozen
- **Language Model**: Pre-trained Gemma frozen
- **Objective**: Only train the vision-language connector

### Stage 2: Multimodal Pre-training
- **Data**: Web-scale image-text pairs
- **Objective**: Image captioning with prefix LM loss
- **Components Trained**: Connector + Language model
- **Vision Encoder**: Remains frozen

### Stage 3: Task-Specific Fine-tuning
- **Tasks**: VQA, Object Detection, OCR, etc.
- **Data**: Task-specific datasets
- **Components Trained**: All trainable parameters
- **Techniques**: Task-specific prompting and formatting

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 1e-5 (with cosine scheduling)
- **Batch Size**: 256 (across multiple GPUs)
- **Sequence Length**: 512 tokens
- **Image Resolution**: 224×224
- **Training Steps**: ~1M steps

## 📊 Eval & Performance

### Benchmark Results
PaliGemma achieves competitive performance across multiple vision-language benchmarks:

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| Image Captioning | COCO | CIDEr | 113.6 |
| VQA | VQAv2 | Accuracy | 81.2% |
| Object Detection | COCO | mAP | 44.9 |
| OCR | TextVQA | Accuracy | 71.6% |
| General VL | MMMU | Accuracy | 34.9% |

### Key Strengths
- **Efficiency**: 3B parameters vs larger competitors
- **Versatility**: Single model handles multiple tasks
- **Quality**: Strong performance despite smaller size
- **Open Source**: Full model weights and training code available

## 🔍 Implementation Details

### Key Components

1. **SigLIP Vision Encoder** (`src/models/siglip.py`)
   - Implementation of SigLIP vision transformer
   - Handles patch embedding and positional encoding
   - Outputs 256 visual tokens

2. **Gemma Language Model** (`src/models/gemma.py`)
   - Decoder-only transformer implementation
   - RMSNorm and SwiGLU activation
   - Rotary positional embeddings (RoPE)

3. **Vision-Language Connector** (`src/models/connector.py`)
   - Simple linear projection layer
   - Maps vision tokens to language model dimension

4. **Training Loop** (`src/training/trainer.py`)
   - Handles prefix LM attention masking
   - Implements gradient accumulation
   - Supports mixed precision training

### Attention Mechanism
```python
# prefix LM attention pattern
# vision tokens can attend to all previous tokens
# text tokens use causal masking
attention_mask = create_prefix_lm_mask(
    vision_length=256, 
    text_length=sequence_length-256
)
```

## 🎯 Supported Tasks

1. **Image Captioning**
   ```python
   prompt = "caption en"
   response = model.generate(image, prompt)
   ```

2. **Visual Question Answering**
   ```python
   prompt = "What is in this image?"
   response = model.generate(image, prompt)
   ```

3. **Object Detection**
   ```python
   prompt = "detect person ; car"
   response = model.generate(image, prompt) 
   ```

4. **Optical Character Recognition**
   ```python
   prompt = "What does the text say?"
   response = model.generate(image, prompt)
   ```

## ⚙️ Configuration

### Model Configuration
```yaml
model:
  vision_encoder: "siglip-so400m-patch14-224"
  language_model: "gemma-2b"
  hidden_size: 2048
  vision_tokens: 256
  max_length: 512

training:
  batch_size: 32
  learning_rate: 1e-5
  num_epochs: 10
  warmup_steps: 1000
```
## 📚 References

1. **PaliGemma Paper**: [arXiv:2407.07726](https://arxiv.org/abs/2407.07726)
2. **SigLIP Paper**: [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
3. **Gemma Paper**: [arXiv:2403.08295](https://arxiv.org/abs/2403.08295)
4. **Big-Vision**: [Google Research GitHub](https://github.com/google-research/big_vision)

