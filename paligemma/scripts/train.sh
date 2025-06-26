#!/bin/bash

# PaliGemma Training Script
# Usage: ./scripts/train.sh [config_path] [additional_args...]

set -e  # Exit on any error

# Default configuration
CONFIG_PATH="${1:-configs/config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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
    echo "Usage: $0 [config_path] [additional_args...]"
    exit 1
fi

log "Starting PaliGemma training pipeline..."
log "Configuration file: $CONFIG_PATH"
log "Project root: $PROJECT_ROOT"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
log "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
log "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    warning "requirements.txt not found. Installing basic dependencies..."
    pip install torch torchvision transformers accelerate tensorboard pyyaml pillow tqdm matplotlib
fi

# Check GPU availability
log "Checking hardware..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('Running on CPU')
"

# Create necessary directories
log "Creating output directories..."
mkdir -p outputs logs checkpoints cache

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Check if using Hugging Face Hub (for model downloads)
if [ -n "$HF_TOKEN" ]; then
    log "Using Hugging Face token for model access"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# Parse additional arguments
PYTHON_ARGS=""
shift || true  # Remove config path from arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            PYTHON_ARGS="$PYTHON_ARGS --resume_from_checkpoint $2"
            shift 2
            ;;
        --debug)
            PYTHON_ARGS="$PYTHON_ARGS --debug"
            export CUDA_LAUNCH_BLOCKING=1
            shift
            ;;
        --profile)
            PYTHON_ARGS="$PYTHON_ARGS --profile"
            shift
            ;;
        --dry-run)
            PYTHON_ARGS="$PYTHON_ARGS --dry_run"
            shift
            ;;
        *)
            PYTHON_ARGS="$PYTHON_ARGS $1"
            shift
            ;;
    esac
done

# Start training
log "Starting training with configuration: $CONFIG_PATH"
log "Additional arguments: $PYTHON_ARGS"

# Run training with error handling
if python3 -m src.training.train --config "$CONFIG_PATH" $PYTHON_ARGS; then
    success "Training completed successfully!"
else
    error "Training failed with exit code $?"
    exit 1
fi

# Optional: Run evaluation after training
read -p "Run evaluation on test set? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Running evaluation..."
    bash "$SCRIPT_DIR/evaluate.sh" "$CONFIG_PATH"
fi

# Cleanup
log "Cleaning up temporary files..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --gpu-reset || true
fi

success "Training pipeline completed!"
log "Check outputs/ directory for saved models and logs/ for training logs"