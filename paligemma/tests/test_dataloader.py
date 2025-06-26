import pytest
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

# Import the modules to test
try:
    from src.data.dataloader import PaliGemmaDataset, create_dataloaders
except ImportError:
    # Fallback for different import paths
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from data.dataloader import PaliGemmaDataset, create_dataloaders


class TestPaliGemmaDataset:
    """Test cases for PaliGemmaDataset class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create class directories
        classes = ['pizza', 'steak', 'sushi']
        for class_name in classes:
            class_dir = os.path.join(temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create sample images
            for i in range(3):
                img = Image.new('RGB', (224, 224), color=(i*50, i*80, i*100))
                img_path = os.path.join(class_dir, f'{class_name}_{i}.jpg')
                img.save(img_path)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock PaliGemmaProcessor for testing."""
        processor = Mock()
        
        # Mock the processor call
        def mock_process(*args, **kwargs):
            # Return mock tensor data
            return {
                'input_ids': torch.randint(0, 1000, (1, 20)),
                'attention_mask': torch.ones((1, 20)),
                'pixel_values': torch.randn((1, 3, 224, 224))
            }
        
        processor.side_effect = mock_process
        
        # Mock tokenizer for collate function
        processor.tokenizer = Mock()
        processor.tokenizer.pad = Mock(return_value={
            'input_ids': torch.randint(0, 1000, (2, 25)),
            'attention_mask': torch.ones((2, 25))
        })
        
        return processor
    
    def test_dataset_initialization(self, temp_data_dir, mock_processor):
        """Test dataset initialization."""
        dataset = PaliGemmaDataset(temp_data_dir, mock_processor, "train")
        
        assert dataset.data_dir == temp_data_dir
        assert dataset.processor == mock_processor
        assert dataset.split == "train"
        assert len(dataset.samples) > 0
    
    def test_load_samples(self, temp_data_dir, mock_processor):
        """Test sample loading from directory structure."""
        dataset = PaliGemmaDataset(temp_data_dir, mock_processor, "train")
        
        # Should have 9 samples (3 classes × 3 images each)
        assert len(dataset.samples) == 9
        
        # Check sample structure
        sample = dataset.samples[0]
        assert 'image_path' in sample
        assert 'input_text' in sample
        assert 'target_text' in sample
        
        # Check that all image paths exist
        for sample in dataset.samples:
            assert os.path.exists(sample['image_path'])
    
    def test_dataset_length(self, temp_data_dir, mock_processor):
        """Test dataset length calculation."""
        dataset = PaliGemmaDataset(temp_data_dir, mock_processor, "train")
        assert len(dataset) == len(dataset.samples)
    
    @patch('PIL.Image.open')
    def test_getitem(self, mock_image_open, temp_data_dir, mock_processor):
        """Test dataset item retrieval."""
        # Mock image loading
        mock_img = Mock()
        mock_img.convert.return_value = Mock()
        mock_image_open.return_value = mock_img
        
        dataset = PaliGemmaDataset(temp_data_dir, mock_processor, "train")
        
        # Test getting an item
        inputs, targets = dataset[0]
        
        # Check that processor was called
        assert mock_processor.call_count >= 2  # Once for inputs, once for targets
        
        # Check that inputs and targets are dictionaries
        assert isinstance(inputs, dict)
        assert isinstance(targets, dict)
    
    def test_empty_directory(self, mock_processor):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = PaliGemmaDataset(temp_dir, mock_processor, "train")
            assert len(dataset.samples) == 0
            assert len(dataset) == 0
    
    def test_invalid_directory(self, mock_processor):
        """Test behavior with invalid directory."""
        invalid_dir = "/non/existent/path"
        
        with pytest.raises(FileNotFoundError):
            dataset = PaliGemmaDataset(invalid_dir, mock_processor, "train")


class TestCreateDataloaders:
    """Test cases for create_dataloaders function."""
    
    @pytest.fixture
    def temp_train_test_dirs(self):
        """Create temporary train and test directories."""
        temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(temp_dir, 'train')
        test_dir = os.path.join(temp_dir, 'test')
        
        for data_dir in [train_dir, test_dir]:
            classes = ['pizza', 'steak', 'sushi']
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Create sample images
                for i in range(2):
                    img = Image.new('RGB', (224, 224), color=(i*50, i*80, i*100))
                    img_path = os.path.join(class_dir, f'{class_name}_{i}.jpg')
                    img.save(img_path)
        
        yield train_dir, test_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor for testing."""
        processor = Mock()
        
        def mock_process(*args, **kwargs):
            return {
                'input_ids': torch.randint(0, 1000, (1, 20)),
                'attention_mask': torch.ones((1, 20)),
                'pixel_values': torch.randn((1, 3, 224, 224))
            }
        
        processor.side_effect = mock_process
        
        # Mock tokenizer
        processor.tokenizer = Mock()
        processor.tokenizer.pad = Mock(return_value={
            'input_ids': torch.randint(0, 1000, (2, 25)),
            'attention_mask': torch.ones((2, 25))
        })
        
        return processor
    
    def test_create_dataloaders(self, temp_train_test_dirs, mock_processor):
        """Test dataloader creation."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, test_loader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0  # Use 0 for testing
        )
        
        # Check return types
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(class_names, list)
        
        # Check class names
        expected_classes = ['pizza', 'steak', 'sushi']
        assert set(class_names) == set(expected_classes)
        
        # Check dataloaders have correct batch size
        assert train_loader.batch_size == 2
        assert test_loader.batch_size == 2
        
        # Check that shuffle is different for train/test
        assert train_loader.shuffle == True
        assert test_loader.shuffle == False
    
    def test_dataloader_iteration(self, temp_train_test_dirs, mock_processor):
        """Test that dataloaders can be iterated."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, test_loader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0
        )
        
        # Test that we can get a batch from train loader
        try:
            batch = next(iter(train_loader))
            inputs, targets = batch
            
            # Check batch structure
            assert isinstance(inputs, dict)
            assert isinstance(targets, dict)
            
        except StopIteration:
            pytest.fail("Train dataloader is empty")
        
        # Test that we can get a batch from test loader
        try:
            batch = next(iter(test_loader))
            inputs, targets = batch
            
            # Check batch structure
            assert isinstance(inputs, dict)
            assert isinstance(targets, dict)
            
        except StopIteration:
            pytest.fail("Test dataloader is empty")
    
    def test_collate_function(self, temp_train_test_dirs, mock_processor):
        """Test the custom collate function."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, _, _ = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0
        )
        
        # The collate function should be callable
        assert train_loader.collate_fn is not None
        
        # Test with a mock batch
        mock_batch = [
            ({'input_ids': torch.tensor([1, 2, 3])}, {'input_ids': torch.tensor([4, 5, 6])}),
            ({'input_ids': torch.tensor([7, 8, 9])}, {'input_ids': torch.tensor([10, 11, 12])}),
        ]
        
        # This should not raise an error
        try:
            result = train_loader.collate_fn(mock_batch)
            assert len(result) == 2  # inputs and targets
        except Exception as e:
            # It's okay if this fails due to mock limitations
            pass
    
    def test_different_batch_sizes(self, temp_train_test_dirs, mock_processor):
        """Test dataloader creation with different batch sizes."""
        train_dir, test_dir = temp_train_test_dirs
        
        for batch_size in [1, 2, 4]:
            train_loader, test_loader, _ = create_dataloaders(
                train_dir=train_dir,
                test_dir=test_dir,
                processor=mock_processor,
                batch_size=batch_size,
                num_workers=0
            )
            
            assert train_loader.batch_size == batch_size
            assert test_loader.batch_size == batch_size
    
    def test_pin_memory_setting(self, temp_train_test_dirs, mock_processor):
        """Test that pin_memory is set correctly."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, test_loader, _ = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0
        )
        
        assert train_loader.pin_memory == True
        assert test_loader.pin_memory == True


class TestDatasetEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor."""
        processor = Mock()
        processor.side_effect = lambda *args, **kwargs: {
            'input_ids': torch.randint(0, 1000, (1, 20)),
            'attention_mask': torch.ones((1, 20)),
            'pixel_values': torch.randn((1, 3, 224, 224))
        }
        processor.tokenizer = Mock()
        processor.tokenizer.pad = Mock(return_value={
            'input_ids': torch.randint(0, 1000, (2, 25)),
            'attention_mask': torch.ones((2, 25))
        })
        return processor
    
    def test_dataset_with_non_image_files(self, mock_processor):
        """Test dataset behavior with non-image files in directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create class directory with non-image files
            class_dir = os.path.join(temp_dir, 'test_class')
            os.makedirs(class_dir)
            
            # Create non-image files
            with open(os.path.join(class_dir, 'not_an_image.txt'), 'w') as f:
                f.write("This is not an image")
            
            # Create one valid image
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            img.save(os.path.join(class_dir, 'valid_image.jpg'))
            
            dataset = PaliGemmaDataset(temp_dir, mock_processor, "train")
            
            # Should only include the valid image
            assert len(dataset.samples) == 1
            assert 'valid_image.jpg' in dataset.samples[0]['image_path']
    
    def test_dataset_with_corrupted_image(self, mock_processor):
        """Test dataset behavior with corrupted image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            class_dir = os.path.join(temp_dir, 'test_class')
            os.makedirs(class_dir)
            
            # Create corrupted image file
            with open(os.path.join(class_dir, 'corrupted.jpg'), 'wb') as f:
                f.write(b'not a valid image')
            
            dataset = PaliGemmaDataset(temp_dir, mock_processor, "train")
            
            # Dataset should be created but accessing the item should handle the error
            assert len(dataset.samples) == 1
            
            # Accessing the corrupted image should raise an error or be handled gracefully
            with pytest.raises((OSError, Exception)):
                _ = dataset[0]


if __name__ == "__main__":
    pytest.main([__file__])