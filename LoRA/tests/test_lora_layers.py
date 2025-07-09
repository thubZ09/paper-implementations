import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lora_layers import LoRALayer, LoRALinear, LoRAConv2d


class TestLoRALayers(unittest.TestCase):
    def setUp(self):
        """set up test fixtures before each test method"""
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 64
        self.output_dim = 32
        self.rank = 8
        self.alpha = 16
        self.dropout = 0.1
        
    def test_lora_layer_initialization(self):
        """test LoRALayer initialization"""
        lora_layer = LoRALayer(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        self.assertEqual(lora_layer.rank, self.rank)
        self.assertEqual(lora_layer.alpha, self.alpha)
        self.assertEqual(lora_layer.scaling, self.alpha / self.rank)
        self.assertIsInstance(lora_layer.dropout, nn.Dropout)
        
    def test_lora_linear_initialization(self):
        """test LoRALinear layer initialization"""
        lora_linear = LoRALinear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        #check original linear layer
        self.assertEqual(lora_linear.linear.in_features, self.hidden_dim)
        self.assertEqual(lora_linear.linear.out_features, self.output_dim)
        
        #check LoRA matrices
        self.assertEqual(lora_linear.lora_A.shape, (self.rank, self.hidden_dim))
        self.assertEqual(lora_linear.lora_B.shape, (self.output_dim, self.rank))
        
        #check that original weights are frozen
        self.assertFalse(lora_linear.linear.weight.requires_grad)
        if lora_linear.linear.bias is not None:
            self.assertFalse(lora_linear.linear.bias.requires_grad)
            
        #check that LoRA weights require gradients
        self.assertTrue(lora_linear.lora_A.requires_grad)
        self.assertTrue(lora_linear.lora_B.requires_grad)
        
    def test_lora_linear_forward(self):
        """test LoRALinear forward pass"""
        lora_linear = LoRALinear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        #test input
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        #forward pass
        output = lora_linear(x)
        
        #check output shape
        expected_shape = (self.batch_size, self.seq_len, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        
        #check that output is different from original linear layer
        original_output = lora_linear.linear(x)
        self.assertFalse(torch.allclose(output, original_output))
        
    def test_lora_conv2d_initialization(self):
        """test LoRAConv2d layer initialization"""
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        
        lora_conv = LoRAConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        #check original conv layer
        self.assertEqual(lora_conv.conv.in_channels, in_channels)
        self.assertEqual(lora_conv.conv.out_channels, out_channels)
        self.assertEqual(lora_conv.conv.kernel_size, (kernel_size, kernel_size))
        
        #check LoRA matrices
        self.assertEqual(lora_conv.lora_A.shape, (self.rank, in_channels, kernel_size, kernel_size))
        self.assertEqual(lora_conv.lora_B.shape, (out_channels, self.rank, 1, 1))
        
        #check that original weights are frozen
        self.assertFalse(lora_conv.conv.weight.requires_grad)
        if lora_conv.conv.bias is not None:
            self.assertFalse(lora_conv.conv.bias.requires_grad)
            
    def test_lora_conv2d_forward(self):
        """test LoRAConv2d forward pass"""
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        height, width = 32, 32
        
        lora_conv = LoRAConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        #test input
        x = torch.randn(self.batch_size, in_channels, height, width)
        
        #forward pass
        output = lora_conv(x)
        
        #check output shape (considering padding)
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1
        expected_shape = (self.batch_size, out_channels, expected_height, expected_width)
        self.assertEqual(output.shape, expected_shape)
        
    def test_lora_parameter_counting(self):
        """test that LoRA reduces the number of trainable parameters."""
        in_features = 1000
        out_features = 1000
        rank = 16
        
        #original linear layer
        original_linear = nn.Linear(in_features, out_features)
        original_params = sum(p.numel() for p in original_linear.parameters() if p.requires_grad)
        
        #LoRA linear layer
        lora_linear = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=32,
            dropout=0.1
        )
        lora_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
        
        self.assertLess(lora_params, original_params)
        
        expected_lora_params = rank * (in_features + out_features)
        self.assertEqual(lora_params, expected_lora_params)
        
    def test_lora_enable_disable(self):
        """test enabling and disabling LoRA"""
        lora_linear = LoRALinear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        #test with LoRA enabled (default)
        output_with_lora = lora_linear(x)
        
        #disable LoRA
        lora_linear.enable_lora = False
        output_without_lora = lora_linear(x)
        
        #re-enable LoRA
        lora_linear.enable_lora = True
        output_with_lora_again = lora_linear(x)
        
        self.assertFalse(torch.allclose(output_with_lora, output_without_lora))       
        self.assertTrue(torch.allclose(output_with_lora, output_with_lora_again))
        
    def test_lora_zero_rank(self):
        """test LoRA with zero rank (should behave like original layer)"""
        lora_linear = LoRALinear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            rank=0,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        #forward pass
        output = lora_linear(x)
        original_output = lora_linear.linear(x)
        
        #should be identical to original linear layer
        self.assertTrue(torch.allclose(output, original_output))


if __name__ == '__main__':
    unittest.main()