#trainer.py
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PaliGemma model for a single epoch.

    Args:
        model: A PaliGemma model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy).
    """
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Send data to target device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # 1. Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = targets['input_ids'].view(-1)
        
        # Ignore padding tokens in loss calculation
        mask = labels != -100  # -100 is typically used for ignored tokens
        if mask.any():
            logits = logits[mask]
            labels = labels[mask]
        
        # 2. Calculate loss
        loss = torch.nn.functional.cross_entropy(logits, labels)
        train_loss += loss.item()
        
        # Calculate accuracy (optional, for monitoring)
        with torch.no_grad():
            predicted = torch.argmax(logits, dim=-1)
            correct = (predicted == labels).float()
            accuracy = correct.mean()
            train_acc += accuracy.item()
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Memory cleanup for large models
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Adjust metrics to get average loss and accuracy per epoch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PaliGemma model for a single epoch.

    Args:
        model: A PaliGemma model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy).
    """
    # Put model in eval mode
    model.eval()
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Send data to target device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # 1. Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            labels = targets['input_ids'].view(-1)
            
            # Ignore padding tokens in loss calculation
            mask = labels != -100
            if mask.any():
                logits = logits[mask]
                labels = labels[mask]
            
            # 2. Calculate loss
            loss = torch.nn.functional.cross_entropy(logits, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)
            correct = (predicted == labels).float()
            accuracy = correct.mean()
            test_acc += accuracy.item()
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer=None) -> Dict[str, List]:
    """Trains and tests a PaliGemma model.

    Args:
        model: A PaliGemma model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        writer: Optional tensorboard writer for logging.

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics.
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), desc="Training"):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        # Testing phase
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        # Print results
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # Log to tensorboard if writer provided
        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "test_loss": test_loss
                },
                global_step=epoch
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "train_acc": train_acc,
                    "test_acc": test_acc
                },
                global_step=epoch
            )
        
        # Memory cleanup after each epoch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Close writer if provided
    if writer:
        writer.close()
    
    return results