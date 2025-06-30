import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.trainer import train_step, test_step, train


class MockPaliGemmaModel(nn.Module):
    """Mock PaliGemma model for testing purposes."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, **kwargs):
        batch_size = kwargs.get('input_ids', kwargs.get('pixel_values')).shape[0]
        seq_len = 10  #fixed sequence length for testing
        
        #create mock logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        outputs = Mock()
        outputs.logits = logits
        return outputs


class TestTrainingModule:
    """Test suite for the training module."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock PaliGemma model."""
        return MockPaliGemmaModel()
    
    @pytest.fixture
    def mock_optimizer(self, mock_model):
        """Create a mock optimizer."""
        return torch.optim.Adam(mock_model.parameters(), lr=0.001)
    
    @pytest.fixture
    def mock_loss_fn(self):
        """Create a mock loss function."""
        return nn.CrossEntropyLoss(ignore_index=-100)
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader with proper structure."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        inputs_list = []
        targets_list = []
        
        for _ in range(3): 
            inputs = {
                'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'pixel_values': torch.randn(batch_size, 3, 224, 224)
            }
            targets = {
                'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len))
            }
            inputs_list.append((inputs, targets))
        
        return inputs_list
    
    def test_train_step_basic_functionality(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test basic functionality of train_step."""
        mock_model.to(device)
        
        train_loss, train_acc = train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        #values are returnedChec
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1
    
    def test_train_step_model_in_training_mode(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test that model is set to training mode during train_step."""
        mock_model.eval()  
        assert not mock_model.training
        
        train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        assert mock_model.training
    
    def test_train_step_gradient_computation(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test that gradients are computed during train_step."""
        initial_grads = [param.grad for param in mock_model.parameters()]
        
        train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        #checking gradients have been computed
        final_grads = [param.grad for param in mock_model.parameters()]
        
        assert any(grad is not None for grad in final_grads)
    
    def test_test_step_basic_functionality(self, mock_model, mock_loss_fn, mock_dataloader, device):
        """Test basic functionality of test_step."""
        mock_model.to(device)
        
        test_loss, test_acc = test_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            device=device
        )
        
        assert isinstance(test_loss, float)
        assert isinstance(test_acc, float)
        assert test_loss >= 0
        assert 0 <= test_acc <= 1
    
    def test_test_step_model_in_eval_mode(self, mock_model, mock_loss_fn, mock_dataloader, device):
        """Test that model is set to eval mode during test_step."""
        mock_model.train()  
        assert mock_model.training
        
        test_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            device=device
        )
        
        assert not mock_model.training
    
    def test_test_step_no_gradient_computation(self, mock_model, mock_loss_fn, mock_dataloader, device):
        """Test that no gradients are computed during test_step."""
        #enable gradient computation initially
        torch.set_grad_enabled(True)
        
        with patch('torch.inference_mode') as mock_inference_mode:
            mock_context = MagicMock()
            mock_inference_mode.return_value.__enter__ = Mock(return_value=mock_context)
            mock_inference_mode.return_value.__exit__ = Mock(return_value=None)
            
            test_step(
                model=mock_model,
                dataloader=mock_dataloader,
                loss_fn=mock_loss_fn,
                device=device
            )
            
            mock_inference_mode.assert_called_once()
    
    def test_train_function_basic_functionality(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """test basic functionality of the train function."""
        epochs = 2
        
        results = train(
            model=mock_model,
            train_dataloader=mock_dataloader,
            test_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            loss_fn=mock_loss_fn,
            epochs=epochs,
            device=device
        )
        
        #check results structure
        assert isinstance(results, dict)
        assert "train_loss" in results
        assert "train_acc" in results
        assert "test_loss" in results
        assert "test_acc" in results
        
        #heck that results have correct length
        assert len(results["train_loss"]) == epochs
        assert len(results["train_acc"]) == epochs
        assert len(results["test_loss"]) == epochs
        assert len(results["test_acc"]) == epochs
    
    def test_train_function_with_tensorboard_writer(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test train function with tensorboard writer."""
        epochs = 1
        mock_writer = Mock()
        
        results = train(
            model=mock_model,
            train_dataloader=mock_dataloader,
            test_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            loss_fn=mock_loss_fn,
            epochs=epochs,
            device=device,
            writer=mock_writer
        )
        
        assert mock_writer.add_scalars.call_count >= epochs
        mock_writer.close.assert_called_once()
    
    def test_train_function_model_moved_to_device(self, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test that model is moved to correct device during training."""
        mock_model = MockPaliGemmaModel()
        
        original_to = mock_model.to
        mock_model.to = Mock(side_effect=original_to)
        
        train(
            model=mock_model,
            train_dataloader=mock_dataloader,
            test_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            loss_fn=mock_loss_fn,
            epochs=1,
            device=device
        )
        
        mock_model.to.assert_called_with(device)
    
    def test_masked_loss_calculation(self, mock_model, mock_optimizer, mock_loss_fn, device):
        """Test loss calculation with masked tokens."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        inputs = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'pixel_values': torch.randn(batch_size, 3, 224, 224)
        }
        
        #create targets with some -100 (ignored) tokens
        targets = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len))
        }
        targets['input_ids'][0, :3] = -100  #mask first 3 tokens of first example
        
        mock_dataloader = [(inputs, targets)]
        
        train_loss, train_acc = train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
    
    def test_empty_dataloader_handling(self, mock_model, mock_optimizer, mock_loss_fn, device):
        """Test handling of empty dataloader."""
        empty_dataloader = []
        
        train_loss, train_acc = train_step(
            model=mock_model,
            dataloader=empty_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        assert train_loss == 0
        assert train_acc == 0
    
    def test_memory_cleanup_called(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test that memory cleanup is called during training."""
        with patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.is_available', return_value=True):
            
            train_step(
                model=mock_model,
                dataloader=mock_dataloader,
                loss_fn=mock_loss_fn,
                optimizer=mock_optimizer,
                device=device
            )
            
            assert mock_empty_cache.call_count > 0
    
    def test_different_batch_sizes(self, mock_model, mock_optimizer, mock_loss_fn, device):
        """Test training with different batch sizes."""
        vocab_size = 1000
        seq_len = 10
        
        for batch_size in [1, 2, 4]:
            inputs = {
                'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'pixel_values': torch.randn(batch_size, 3, 224, 224)
            }
            targets = {
                'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len))
            }
            
            mock_dataloader = [(inputs, targets)]
            
            train_loss, train_acc = train_step(
                model=mock_model,
                dataloader=mock_dataloader,
                loss_fn=mock_loss_fn,
                optimizer=mock_optimizer,
                device=device
            )
            
            assert isinstance(train_loss, float)
            assert isinstance(train_acc, float)
    
    def test_optimizer_step_called(self, mock_model, mock_loss_fn, mock_dataloader, device):
        """Test that optimizer step is called during training."""
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        assert mock_optimizer.zero_grad.call_count >= len(mock_dataloader)
        assert mock_optimizer.step.call_count >= len(mock_dataloader)
    
    def test_loss_function_called_correctly(self, mock_model, mock_optimizer, mock_dataloader, device):
        """Test that loss function is called with correct arguments."""
        mock_loss_fn = Mock(return_value=torch.tensor(1.0))
        
        train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        assert mock_loss_fn.call_count >= len(mock_dataloader)
    
    @pytest.mark.parametrize("epochs", [1, 2, 5])
    def test_train_function_different_epochs(self, mock_model, mock_optimizer, mock_loss_fn, mock_dataloader, device, epochs):
        """Test train function with different numbers of epochs."""
        results = train(
            model=mock_model,
            train_dataloader=mock_dataloader,
            test_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            loss_fn=mock_loss_fn,
            epochs=epochs,
            device=device
        )
        
        for key in ["train_loss", "train_acc", "test_loss", "test_acc"]:
            assert len(results[key]) == epochs
    
    def test_error_handling_in_train_step(self, mock_optimizer, mock_loss_fn, mock_dataloader, device):
        """Test error handling when model forward pass fails."""
        class BrokenModel(nn.Module):
            def forward(self, **kwargs):
                raise RuntimeError("Model forward pass failed")
        
        broken_model = BrokenModel()
        
        #raising an exception
        with pytest.raises(RuntimeError, match="Model forward pass failed"):
            train_step(
                model=broken_model,
                dataloader=mock_dataloader,
                loss_fn=mock_loss_fn,
                optimizer=mock_optimizer,
                device=device
            )
    
    def test_all_mask_tokens_handling(self, mock_model, mock_optimizer, mock_loss_fn, device):
        """Test handling when all tokens are masked."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        inputs = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'pixel_values': torch.randn(batch_size, 3, 224, 224)
        }
        
        targets = {
            'input_ids': torch.full((batch_size, seq_len), -100)
        }
        
        mock_dataloader = [(inputs, targets)]
        
        #handle the case where no tokens are used for loss
        train_loss, train_acc = train_step(
            model=mock_model,
            dataloader=mock_dataloader,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=device
        )
        
        #should return some values even with all masked tokens
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)


if __name__ == "__main__":
    #run tests if script is executed directly
    pytest.main([__file__, "-v"])