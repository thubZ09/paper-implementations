import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lora_model import LoRAModel, apply_lora_to_model


class TestLoRAModel(unittest.TestCase):
    def setUp(self):
        """set up test fixtures before each test method"""
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 1000
        self.hidden_dim = 128
        self.num_layers = 4
        self.num_heads = 8
        self.rank = 8
        self.alpha = 16
        self.dropout = 0.1
        
    def create_simple_transformer(self):
        """ceate a simple transformer model for testing"""
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=num_layers
                )
                self.output_head = nn.Linear(hidden_dim, vocab_size)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output_head(x)
                
        return SimpleTransformer(vocab_size, hidden_dim, num_layers, num_heads)
        
    def test_lora_model_initialization(self):
        """test LoRAModel initialization"""
        base_model = self.create_simple_transformer()
        
        lora_model = LoRAModel(
            base_model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        self.assertEqual(lora_model.rank, self.rank)
        self.assertEqual(lora_model.alpha, self.alpha)
        self.assertEqual(lora_model.dropout, self.dropout)
        self.assertIsInstance(lora_model.base_model, nn.Module)
        
    def test_lora_model_forward(self):
        """test LoRAModel forward pass"""
        base_model = self.create_simple_transformer()
        
        lora_model = LoRAModel(
            base_model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #test input
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        #forward pass
        output = lora_model(x)
        
        #check output shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_apply_lora_to_model(self):
        """test applying LoRA to a model"""
        base_model = self.create_simple_transformer()
        
        #count original param
        original_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        
        #apply LoRA
        lora_model = apply_lora_to_model(
            model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #count LoRA parameters (only trainable ones)
        lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        #lora should have fewer trainable parameters
        self.assertLess(lora_params, original_params)
        
    def test_lora_save_load_adapter(self):
        """test saving and loading LoRA adapter weights"""
        base_model = self.create_simple_transformer()
        
        lora_model = LoRAModel(
            base_model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #get initial adapter weights
        initial_weights = lora_model.get_lora_state_dict()
        
        #modify some weights
        with torch.no_grad():
            for name, param in lora_model.named_parameters():
                if 'lora' in name and param.requires_grad:
                    param.fill_(1.0)
        
        #get modified weights
        modified_weights = lora_model.get_lora_state_dict()
        
        #check that weights are different
        for key in initial_weights:
            self.assertFalse(torch.allclose(initial_weights[key], modified_weights[key]))
        
        #load initial weights back
        lora_model.load_lora_state_dict(initial_weights)
        
        #get loaded weights
        loaded_weights = lora_model.get_lora_state_dict()
        
        #ceck that weights are restored
        for key in initial_weights:
            self.assertTrue(torch.allclose(initial_weights[key], loaded_weights[key]))
            
    def test_lora_enable_disable_adapters(self):
        """test enabling and disabling LoRA adapters"""
        base_model = self.create_simple_transformer()
        
        lora_model = LoRAModel(
            base_model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #test input
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        #forward pass with LoRA enabled
        output_with_lora = lora_model(x)
        
        #disable LoRA
        lora_model.enable_lora_adapters(False)
        output_without_lora = lora_model(x)
        
        #re-enable LoRA
        lora_model.enable_lora_adapters(True)
        output_with_lora_again = lora_model(x)
        
        self.assertFalse(torch.allclose(output_with_lora, output_without_lora))        
        self.assertTrue(torch.allclose(output_with_lora, output_with_lora_again))
        
    def test_lora_parameter_efficiency(self):
        """test that LoRA is parameter efficient"""
        base_model = self.create_simple_transformer()
        
        #count all parameters in base model
        base_total_params = sum(p.numel() for p in base_model.parameters())
        
        #apply LoRA
        lora_model = apply_lora_to_model(
            model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #count trainable parameters (only LoRA adapters)
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        #LoRA should use much fewer parameters
        efficiency_ratio = trainable_params / base_total_params
        self.assertLess(efficiency_ratio, 0.1) 
        
    def test_lora_gradient_flow(self):
        """test that gradients only flow through LoRA parameters."""
        base_model = self.create_simple_transformer()
        
        lora_model = LoRAModel(
            base_model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #test input and target
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        #forward pass
        output = lora_model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), target.view(-1))
        
        #backward pass
        loss.backward()
        
        #check that only LoRA parameters have gradients
        for name, param in lora_model.named_parameters():
            if 'lora' in name:
                self.assertIsNotNone(param.grad)
            else:
                #base model parameters should not have gradients
                if param.requires_grad:
                    self.assertIsNone(param.grad)
                    
    def test_lora_merge_unmerge(self):
        """test merging and unmerging LoRA weights."""
        base_model = self.create_simple_transformer()
        
        lora_model = LoRAModel(
            base_model=base_model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        
        #test input
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        #forward pass before merging
        output_before_merge = lora_model(x)
        lora_model.merge_lora_weights()
        
        #forward pass after merging (should be same result)
        output_after_merge = lora_model(x)
        self.assertTrue(torch.allclose(output_before_merge, output_after_merge, atol=1e-6))
        
        #unmerge weights
        lora_model.unmerge_lora_weights()
        
        #forward pass after unmerging
        output_after_unmerge = lora_model(x)
        self.assertTrue(torch.allclose(output_before_merge, output_after_unmerge, atol=1e-6))


if __name__ == '__main__':
    unittest.main()