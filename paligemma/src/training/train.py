#train.py
import os
import torch
import data_setup, engine, model_builder, utils
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

#hyperparameters (T4 GPU) 
NUM_EPOCHS = 3  
BATCH_SIZE = 4  
LEARNING_RATE = 1e-5  
MODEL_NAME = "google/paligemma-3b-pt-224"  

train_dir = "data/pizza_steak_sushi/train"  
test_dir = "data/pizza_steak_sushi/test"    

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

print("Loading PaliGemma processor...")
try:
    processor = PaliGemmaProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading processor: {e}")
    print("Please ensure you have access to the PaliGemma model.")
    exit(1)

print("Creating data loaders...")
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    processor=processor,
    batch_size=BATCH_SIZE,
    num_workers=2 
)

print(f"Classes found: {class_names}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")

#create model with help from model_builder.py
print("Loading PaliGemma model...")
try:
    model = model_builder.create_paligemma_model(
        model_name=MODEL_NAME,
        freeze_backbone=True 
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load model directly...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

#move model to device
model = model.to(device)

if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

print("Setting up optimizer...")

backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'vision_tower' in name or 'language_model' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  #lower LR for backbone
    {'params': head_params, 'lr': LEARNING_RATE}  #higher LR for new layers
], weight_decay=0.01)

#set loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

writer = SummaryWriter(log_dir="runs/paligemma_experiment")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

#lr scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=NUM_EPOCHS,
    eta_min=LEARNING_RATE * 0.01
)

print("Starting training...")
print("=" * 50)

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
    
    #update lr
    scheduler.step()
    
    print("Training completed successfully!")
    print("=" * 50)

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
    writer.close()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Saving model...")
try:
    os.makedirs("models", exist_ok=True)
    
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
    
    if hasattr(utils, 'save_model'):
        utils.save_model(
            model=model,
            target_dir="models",
            model_name="paligemma_finetuned_full.pth"
        )

except Exception as e:
    print(f"Error saving model: {e}")

print("Training script completed!")

#optional\
if len(test_dataloader) > 0:
    print("\nRunning sample prediction...")
    try:
        import predictions

        sample_batch = next(iter(test_dataloader))
        sample_inputs, sample_targets = sample_batch
        
        # make a prediction (this would need an actual image path)
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