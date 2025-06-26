#train.py
import os
import torch
import data_setup, engine, model_builder, utils
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# Setup hyperparameters
NUM_EPOCHS = 3  # Reduced for T4 GPU constraints
BATCH_SIZE = 4  # Small batch size for large model
LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
MODEL_NAME = "google/paligemma-3b-pt-224"  # Use the 224px version for T4

# Setup directories
train_dir = "data/pizza_steak_sushi/train"  # Update this to your data path
test_dir = "data/pizza_steak_sushi/test"    # Update this to your data path

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Memory optimization for T4
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

# Initialize processor
print("Loading PaliGemma processor...")
try:
    processor = PaliGemmaProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading processor: {e}")
    print("Please ensure you have access to the PaliGemma model.")
    exit(1)

# Create DataLoaders with help from data_setup.py
print("Creating data loaders...")
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    processor=processor,
    batch_size=BATCH_SIZE,
    num_workers=2  # Reduced for stability
)

print(f"Classes found: {class_names}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")

# Create model with help from model_builder.py
print("Loading PaliGemma model...")
try:
    model = model_builder.create_paligemma_model(
        model_name=MODEL_NAME,
        freeze_backbone=True  # Freeze backbone for faster training on T4
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load model directly...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# Move model to device
model = model.to(device)

# Enable gradient checkpointing for memory efficiency
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Set up optimizer with different learning rates for different parts
print("Setting up optimizer...")

# Separate parameters for different learning rates
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'vision_tower' in name or 'language_model' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

# Create optimizer with different learning rates
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for backbone
    {'params': head_params, 'lr': LEARNING_RATE}  # Higher LR for new layers
], weight_decay=0.01)

# Set loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# Create tensorboard writer
writer = SummaryWriter(log_dir="runs/paligemma_experiment")

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=NUM_EPOCHS,
    eta_min=LEARNING_RATE * 0.01
)

print("Starting training...")
print("=" * 50)

# Start training with help from engine.py
try:
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
        writer=writer
    )
    
    # Update learning rate
    scheduler.step()
    
    print("Training completed successfully!")
    print("=" * 50)
    
    # Print final results
    print("Final Results:")
    print(f"Final train loss: {results['train_loss'][-1]:.4f}")
    print(f"Final test loss: {results['test_loss'][-1]:.4f}")
    if results['train_acc'][-1] > 0:
        print(f"Final train accuracy: {results['train_acc'][-1]:.4f}")
        print(f"Final test accuracy: {results['test_acc'][-1]:.4f}")

except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    writer.close()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Save the model with help from utils.py
print("Saving model...")
try:
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model state dict (more memory efficient)
    model_save_path = "models/paligemma_finetuned.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_results': results,
        'model_name': MODEL_NAME,
        'class_names': class_names,
        'hyperparameters': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        }
    }, model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    
    # Also save just the model for easy loading
    if hasattr(utils, 'save_model'):
        utils.save_model(
            model=model,
            target_dir="models",
            model_name="paligemma_finetuned_full.pth"
        )

except Exception as e:
    print(f"Error saving model: {e}")

print("Training script completed!")

# Optional: Display a sample prediction
if len(test_dataloader) > 0:
    print("\nRunning sample prediction...")
    try:
        import predictions
        
        # Get a sample from test data
        sample_batch = next(iter(test_dataloader))
        sample_inputs, sample_targets = sample_batch
        
        # Make a prediction (this would need an actual image path)
        # predictions.predict_and_plot_image(
        #     model=model,
        #     processor=processor,
        #     image_path="path/to/sample/image.jpg",
        #     prompt="What is in this image?",
        #     device=device
        # )
        
        print("Sample prediction functionality is ready!")
        
    except Exception as e:
        print(f"Could not run sample prediction: {e}")

print("All done!")