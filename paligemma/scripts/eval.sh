#!/bin/bash

# PaliGemma Evaluation Script
# Usage: ./scripts/evaluate.sh [config_path] [model_path] [additional_args...]

set -e

# Default configuration
CONFIG_PATH="${1:-configs/config.yaml}"
MODEL_PATH="${2:-outputs/best_model.pth}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    error "Configuration file not found: $CONFIG_PATH"
    echo "Usage: $0 [config_path] [model_path] [additional_args...]"
    exit 1
fi

log "Starting PaliGemma evaluation pipeline..."
log "Configuration file: $CONFIG_PATH"
log "Model path: $MODEL_PATH"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    warning "Model file not found at: $MODEL_PATH"
    warning "Looking for alternative model files..."
    
    # Try to find model files
    POSSIBLE_MODELS=(
        "outputs/paligemma_finetuned.pth"
        "outputs/best_model.pth"
        "checkpoints/latest.pth"
        "models/paligemma_finetuned.pth"
    )
    
    for model in "${POSSIBLE_MODELS[@]}"; do
        if [ -f "$model" ]; then
            MODEL_PATH="$model"
            log "Found model at: $MODEL_PATH"
            break
        fi
    done
    
    if [ ! -f "$MODEL_PATH" ]; then
        error "No model file found. Please train a model first."
        exit 1
    fi
fi

# Parse additional arguments
PYTHON_ARGS=""
shift 2 || true  # Remove config and model path from arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            PYTHON_ARGS="$PYTHON_ARGS --dataset $2"
            shift 2
            ;;
        --batch-size)
            PYTHON_ARGS="$PYTHON_ARGS --batch_size $2"
            shift 2
            ;;
        --output-dir)
            PYTHON_ARGS="$PYTHON_ARGS --output_dir $2"
            shift 2
            ;;
        --metrics)
            PYTHON_ARGS="$PYTHON_ARGS --metrics $2"
            shift 2
            ;;
        --save-predictions)
            PYTHON_ARGS="$PYTHON_ARGS --save_predictions"
            shift
            ;;
        --verbose)
            PYTHON_ARGS="$PYTHON_ARGS --verbose"
            shift
            ;;
        *)
            PYTHON_ARGS="$PYTHON_ARGS $1"
            shift
            ;;
    esac
done

# Create evaluation output directory
EVAL_OUTPUT_DIR="outputs/evaluation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUTPUT_DIR"

log "Evaluation output directory: $EVAL_OUTPUT_DIR"

# Check GPU availability
log "Checking hardware..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Create evaluation script
cat > evaluate_model.py << 'EOF'
#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.paligemma import create_paligemma_model
from data.dataloader import create_dataloaders
from inference.predict import evaluate_on_dataset
from transformers import PaliGemmaProcessor

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Evaluate PaliGemma model')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='outputs/evaluation', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size for evaluation')
    parser.add_argument('--dataset', help='Dataset split to evaluate on')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['evaluation']['batch_size'] = args.batch_size
    
    print(f"Loading model from: {args.model_path}")
    
    # Initialize processor
    processor = PaliGemmaProcessor.from_pretrained(config['model']['name'])
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # Load from training checkpoint
        model = create_paligemma_model(config['model']['name'])
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', [])
    else:
        # Load full model
        model = checkpoint
        class_names = []
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    # Create dataloaders for evaluation
    # Note: This assumes you have test data in the same format
    # You might need to adjust paths based on your data structure
    try:
        train_dataloader, test_dataloader, _ = create_dataloaders(
            train_dir=config.get('data', {}).get('train_dir', 'data/train'),
            test_dir=config.get('data', {}).get('test_dir', 'data/test'),
            processor=processor,
            batch_size=config['evaluation']['batch_size'],
            num_workers=config['training'].get('dataloader_num_workers', 2)
        )
        
        print(f"Evaluating on {len(test_dataloader)} batches...")
        
        # Run evaluation
        results = evaluate_on_dataset(
            model=model,
            processor=processor,
            dataloader=test_dataloader,
            device=device,
            max_samples=1000
        )
        
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, 'evaluation_results.yaml')
        
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("You may need to adjust data paths in the config file")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
EOF

# Run evaluation
log "Running evaluation..."
if python3 evaluate_model.py --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --output_dir "$EVAL_OUTPUT_DIR" $PYTHON_ARGS; then
    success "Evaluation completed successfully!"
    
    # Show results if they exist
    if [ -f "$EVAL_OUTPUT_DIR/evaluation_results.yaml" ]; then
        log "Evaluation Results:"
        cat "$EVAL_OUTPUT_DIR/evaluation_results.yaml"
    fi
else
    error "Evaluation failed with exit code $?"
    exit 1
fi

# Cleanup temporary files
rm -f evaluate_model.py

# Generate evaluation report
log "Generating evaluation report..."
cat > "$EVAL_OUTPUT_DIR/evaluation_summary.txt" << EOF
PaliGemma Model Evaluation Report
Generated on: $(date)
================================

Configuration: $CONFIG_PATH
Model: $MODEL_PATH
Output Directory: $EVAL_OUTPUT_DIR

Hardware:
- Device: $(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")
$(if command -v nvidia-smi &> /dev/null; then echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"; fi)

Results saved in: $EVAL_OUTPUT_DIR/evaluation_results.yaml
EOF

success "Evaluation pipeline completed!"
log "Results saved in: $EVAL_OUTPUT_DIR"