import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.paligemma import PaliGemmaModel, create_paligemma_model, SimplePaliGemma

class TestPaliGemmaModel:
    """test suite for PaliGemmaModel class."""
    
    @pytest.fixture
    def mock_config(self):
        """mock configuration for testing."""
        mock_config = Mock()
        mock_config.text_config.hidden_size = 2048
        return mock_config
    
    @pytest.fixture
    def mock_transformers_components(self):
        """mock transformers components to avoid loading actual models."""
        with patch('models.paligemma.PaliGemmaForConditionalGeneration') as mock_model, \
             patch('models.paligemma.PaliGemmaConfig') as mock_config:
            
            #setup mock model instance
            mock_instance = Mock()
            mock_instance.config.text_config.hidden_size = 2048
            
            #mck vision tower nd language model for freezing tests
            mock_vision_tower = Mock()
            mock_vision_param = Mock()
            mock_vision_param.requires_grad = True
            mock_vision_tower.parameters.return_value = [mock_vision_param]
            mock_instance.vision_tower = mock_vision_tower
            
            #mock language model layers
            mock_layer = Mock()
            mock_layer_param = Mock()
            mock_layer_param.requires_grad = True
            mock_layer.parameters.return_value = [mock_layer_param]
            
            mock_language_model = Mock()
            mock_language_model.model.layers = [mock_layer, mock_layer, mock_layer]
            mock_instance.language_model = mock_language_model
            
            #outputs
            mock_outputs = Mock()
            mock_outputs.logits = torch.randn(1, 10, 256000)
            mock_instance.return_value = mock_outputs
            mock_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
            
            mock_model.from_pretrained.return_value = mock_instance
            mock_config.from_pretrained.return_value = self.mock_config()
            
            yield mock_model, mock_config, mock_instance
    
    def test_model_initialization_success(self, mock_transformers_components):
        """test successful model initialization."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        model = PaliGemmaModel(model_name="test-model")
        
        assert model.model_name == "test-model"
        assert model.num_classes is None
        assert model.classifier is None
        mock_model.from_pretrained.assert_called_once()
    
    def test_model_initialization_with_classification(self, mock_transformers_components):
        """test model initialization with classification head."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        model = PaliGemmaModel(model_name="test-model", num_classes=10)
        
        assert model.num_classes == 10
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.in_features == 2048
        assert model.classifier.out_features == 10
    
    def test_model_initialization_fallback(self, mock_transformers_components):
        """test model initialization fallback when from_pretrained fails."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        #mke from_pretrained raise an exception
        mock_model.from_pretrained.side_effect = Exception("Model not found")
        
        model = PaliGemmaModel(model_name="test-model")
        
        mock_config.from_pretrained.assert_called_once_with("test-model")
        mock_model.assert_called_once()
    
    def test_freeze_backbone_parameters(self, mock_transformers_components):
        """test freezing backbone parameters."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        model = PaliGemmaModel(model_name="test-model", freeze_backbone=True)
        
        #check that freeze_backbone_parameters was called during init
        #we can verify this by checking if vision tower parameters were accessed
        mock_instance.vision_tower.parameters.assert_called()
        mock_instance.language_model.model.layers.__getitem__.assert_called()
    
    def test_forward_pass_without_classifier(self, mock_transformers_components):
        """Test forward pass without classification head."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        model = PaliGemmaModel(model_name="test-model")
        
        inputs = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
            'pixel_values': torch.randn(1, 3, 224, 224)
        }
        
        outputs = model(inputs)
        
        #should call the underlying model
        mock_instance.assert_called_once_with(**inputs)
        assert outputs is not None
    
    def test_forward_pass_with_classifier(self, mock_transformers_components):
        """Test forward pass with classification head."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        #mock outputs with hidden_states
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(1, 10, 2048)
        mock_outputs.hidden_states = [torch.randn(1, 10, 2048)]
        mock_instance.return_value = mock_outputs
        
        model = PaliGemmaModel(model_name="test-model", num_classes=5)
        
        inputs = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
            'pixel_values': torch.randn(1, 3, 224, 224)
        }
        
        outputs = model(inputs)
        
        #should use classifier
        assert outputs.logits.shape == (1, 5) 
    
    def test_generate_method(self, mock_transformers_components):
        """test generation method."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        model = PaliGemmaModel(model_name="test-model")
        
        inputs = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'pixel_values': torch.randn(1, 3, 224, 224)
        }
        
        generated = model.generate(inputs, max_length=20)
        
        mock_instance.generate.assert_called_once_with(**inputs, max_length=20)
        assert generated is not None
    
    def test_parameter_counting(self, mock_transformers_components):
        """test parameter counting methods."""
        mock_model, mock_config, mock_instance = mock_transformers_components
        
        model = PaliGemmaModel(model_name="test-model")
        
        #mock parameters
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param1.requires_grad = True
        
        mock_param2 = Mock()
        mock_param2.numel.return_value = 200
        mock_param2.requires_grad = False
        
        with patch.object(model, 'parameters', return_value=[mock_param1, mock_param2]):
            total_params = model.get_total_parameters()
            trainable_params = model.get_trainable_parameters()
            
            assert total_params == 300
            assert trainable_params == 100


class TestCreatePaliGemmaModel:
    """Test suite for create_paligemma_model factory function."""
    
    @patch('models.paligemma.PaliGemmaModel')
    def test_factory_function_default_params(self, mock_model_class):
        """Test factory function with default parameters."""
        mock_instance = Mock()
        mock_instance.get_total_parameters.return_value = 1000000
        mock_instance.get_trainable_parameters.return_value = 500000
        mock_model_class.return_value = mock_instance
        
        model = create_paligemma_model()
        
        mock_model_class.assert_called_once_with(
            model_name="google/paligemma-3b-pt-224",
            num_classes=None,
            freeze_backbone=False
        )
        assert model == mock_instance
    
    @patch('models.paligemma.PaliGemmaModel')
    def test_factory_function_custom_params(self, mock_model_class):
        """Test factory function with custom parameters."""
        mock_instance = Mock()
        mock_instance.get_total_parameters.return_value = 1000000
        mock_instance.get_trainable_parameters.return_value = 100000
        mock_model_class.return_value = mock_instance
        
        model = create_paligemma_model(
            model_name="custom-model",
            num_classes=10,
            freeze_backbone=True
        )
        
        mock_model_class.assert_called_once_with(
            model_name="custom-model",
            num_classes=10,
            freeze_backbone=True
        )


class TestSimplePaliGemma:
    """Test suite for SimplePaliGemma wrapper class."""
    
    @patch('models.paligemma.create_paligemma_model')
    def test_simple_wrapper_initialization(self, mock_create_model):
        """Test SimplePaliGemma initialization."""
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        simple_model = SimplePaliGemma(model_name="test-model")
        
        mock_create_model.assert_called_once_with("test-model")
        assert simple_model.paligemma == mock_model
    
    @patch('models.paligemma.create_paligemma_model')
    def test_simple_wrapper_forward(self, mock_create_model):
        """Test SimplePaliGemma forward pass."""
        mock_model = Mock()
        mock_output = torch.randn(1, 10, 1000)
        mock_model.return_value = mock_output
        mock_create_model.return_value = mock_model
        
        simple_model = SimplePaliGemma(model_name="test-model")
        
        inputs = torch.randn(1, 3, 224, 224)
        output = simple_model(inputs)
        
        mock_model.assert_called_once_with(inputs)
        assert torch.equal(output, mock_output)


class TestModelIntegration:
    """Integration tests for the model components."""
    
    def test_model_shapes_consistency(self):
        """Test that model produces consistent output shapes."""
        with patch('models.paligemma.PaliGemmaForConditionalGeneration') as mock_model:
            mock_instance = Mock()
            mock_outputs = Mock()
            mock_outputs.logits = torch.randn(2, 10, 256000)
            mock_instance.return_value = mock_outputs
            mock_instance.config.text_config.hidden_size = 2048
            mock_model.from_pretrained.return_value = mock_instance
            
            model = PaliGemmaModel(model_name="test-model")
            
            batch_size = 2
            seq_length = 10
            
            inputs = {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
                'attention_mask': torch.ones(batch_size, seq_length),
                'pixel_values': torch.randn(batch_size, 3, 224, 224)
            }
            
            outputs = model(inputs)
            
            #check output shape
            expected_shape = (batch_size, seq_length, 256000)
            assert outputs.logits.shape == expected_shape
    
    def test_model_device_handling(self):
        """Test model device handling."""
        with patch('models.paligemma.PaliGemmaForConditionalGeneration') as mock_model:
            mock_instance = Mock()
            mock_instance.config.text_config.hidden_size = 2048
            mock_instance.to.return_value = mock_instance
            mock_model.from_pretrained.return_value = mock_instance
            
            model = PaliGemmaModel(model_name="test-model")
            
            #test moving to device
            device = torch.device('cpu')
            model.to(device)
            
            #verify underlying model was moved to device
            mock_instance.to.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])