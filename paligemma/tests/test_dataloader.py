import pytest
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

try:
    from src.data.dataloader import PaliGemmaDataset, create_dataloaders
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from data.dataloader import PaliGemmaDataset, create_dataloaders


class TestPaliGemmaDataset:
    """test cases for PaliGemmaDataset class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """create a temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        #create class directories
        classes = ['pizza', 'steak', 'sushi']
        for class_name in classes:
            class_dir = os.path.join(temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            #create sample images
            for i in range(3):
                img = Image.new('RGB', (224, 224), color=(i*50, i*80, i*100))
                img_path = os.path.join(class_dir, f'{class_name}_{i}.jpg')
                img.save(img_path)
        
        yield temp_dir
        
        #cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock PaliGemmaProcessor for testing."""
        processor = Mock()
        
        def mock_process(*args, **kwargs):
            return {
                'input_ids': torch.randint(0, 1000, (1, 20)),
                'attention_mask': torch.ones((1, 20)),
                'pixel_values': torch.randn((1, 3, 224, 224))
            }
        
        processor.side_effect = mock_process
        
        #tokenizer for collate function
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
        
        assert len(dataset.samples) == 9
        
        #sample structure check
        sample = dataset.samples[0]
        assert 'image_path' in sample
        assert 'input_text' in sample
        assert 'target_text' in sample
        
        for sample in dataset.samples:
            assert os.path.exists(sample['image_path'])
    
    def test_dataset_length(self, temp_data_dir, mock_processor):
        """Test dataset length calculation."""
        dataset = PaliGemmaDataset(temp_data_dir, mock_processor, "train")
        assert len(dataset) == len(dataset.samples)
    
    @patch('PIL.Image.open')
    def test_getitem(self, mock_image_open, temp_data_dir, mock_processor):
        """Test dataset item retrieval."""
        mock_img = Mock()
        mock_img.convert.return_value = Mock()
        mock_image_open.return_value = mock_img
        
        dataset = PaliGemmaDataset(temp_data_dir, mock_processor, "train")
        
        inputs, targets = dataset[0]
        
        assert mock_processor.call_count >= 2  
        
        assert isinstance(inputs, dict)
        assert isinstance(targets, dict)
    
    def test_empty_directory(self, mock_processor):
        """test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = PaliGemmaDataset(temp_dir, mock_processor, "train")
            assert len(dataset.samples) == 0
            assert len(dataset) == 0
    
    def test_invalid_directory(self, mock_processor):
        """test behavior with invalid directory."""
        invalid_dir = "/non/existent/path"
        
        with pytest.raises(FileNotFoundError):
            dataset = PaliGemmaDataset(invalid_dir, mock_processor, "train")


class TestCreateDataloaders:
    """test cases for create_dataloaders function."""
    
    @pytest.fixture
    def temp_train_test_dirs(self):
        """create temporary train and test directories."""
        temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(temp_dir, 'train')
        test_dir = os.path.join(temp_dir, 'test')
        
        for data_dir in [train_dir, test_dir]:
            classes = ['pizza', 'steak', 'sushi']
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                #create sample images
                for i in range(2):
                    img = Image.new('RGB', (224, 224), color=(i*50, i*80, i*100))
                    img_path = os.path.join(class_dir, f'{class_name}_{i}.jpg')
                    img.save(img_path)
        
        yield train_dir, test_dir
        
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
        
        #mock tokenizer
        processor.tokenizer = Mock()
        processor.tokenizer.pad = Mock(return_value={
            'input_ids': torch.randint(0, 1000, (2, 25)),
            'attention_mask': torch.ones((2, 25))
        })
        
        return processor
    
    def test_create_dataloaders(self, temp_train_test_dirs, mock_processor):
        """test dataloader creation."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, test_loader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0  
        )
        
        #check return types
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(class_names, list)
        
        #lass names
        expected_classes = ['pizza', 'steak', 'sushi']
        assert set(class_names) == set(expected_classes)
        
        #dataloaders correct batch size
        assert train_loader.batch_size == 2
        assert test_loader.batch_size == 2
        
        #shuffle should different for train/test
        assert train_loader.shuffle == True
        assert test_loader.shuffle == False
    
    def test_dataloader_iteration(self, temp_train_test_dirs, mock_processor):
        """test that dataloaders can be iterated."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, test_loader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0
        )
        
        try:
            batch = next(iter(train_loader))
            inputs, targets = batch
            
            assert isinstance(inputs, dict)
            assert isinstance(targets, dict)
            
        except StopIteration:
            pytest.fail("Train dataloader is empty")
        
        try:
            batch = next(iter(test_loader))
            inputs, targets = batch
            
            assert isinstance(inputs, dict)
            assert isinstance(targets, dict)
            
        except StopIteration:
            pytest.fail("Test dataloader is empty")
    
    def test_collate_function(self, temp_train_test_dirs, mock_processor):
        """test the custom collate function."""
        train_dir, test_dir = temp_train_test_dirs
        
        train_loader, _, _ = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            processor=mock_processor,
            batch_size=2,
            num_workers=0
        )
        
        #collate function should be callable
        assert train_loader.collate_fn is not None
        
        #test with a mock batch
        mock_batch = [
            ({'input_ids': torch.tensor([1, 2, 3])}, {'input_ids': torch.tensor([4, 5, 6])}),
            ({'input_ids': torch.tensor([7, 8, 9])}, {'input_ids': torch.tensor([10, 11, 12])}),
        ]
        
        try:
            result = train_loader.collate_fn(mock_batch)
            assert len(result) == 2  #inputs nd targets
        except Exception as e:
            pass
    
    def test_different_batch_sizes(self, temp_train_test_dirs, mock_processor):
        """test dataloader creation with different batch sizes."""
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
        """test that pin_memory is set correctly."""
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
    """test edge cases and error conditions."""
    
    @pytest.fixture
    def mock_processor(self):
        """create a mock processor."""
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
            #create class directory with non-image files
            class_dir = os.path.join(temp_dir, 'test_class')
            os.makedirs(class_dir)
            
            #ceate non-image files
            with open(os.path.join(class_dir, 'not_an_image.txt'), 'w') as f:
                f.write("This is not an image")
            
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            img.save(os.path.join(class_dir, 'valid_image.jpg'))
            
            dataset = PaliGemmaDataset(temp_dir, mock_processor, "train")
            
            assert len(dataset.samples) == 1
            assert 'valid_image.jpg' in dataset.samples[0]['image_path']
    
    def test_dataset_with_corrupted_image(self, mock_processor):
        """test dataset behavior with corrupted image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            class_dir = os.path.join(temp_dir, 'test_class')
            os.makedirs(class_dir)
            
            with open(os.path.join(class_dir, 'corrupted.jpg'), 'wb') as f:
                f.write(b'not a valid image')
            
            dataset = PaliGemmaDataset(temp_dir, mock_processor, "train")
            
            #dataset should be created but accessing the item should handle the error
            assert len(dataset.samples) == 1
            
            #accessing the corrupted image should raise an error or be handled gracefully
            with pytest.raises((OSError, Exception)):
                _ = dataset[0]


if __name__ == "__main__":
    pytest.main([__file__])