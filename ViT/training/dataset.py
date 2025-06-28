from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T

def get_dataset(name, image_size=224, train=True):
    if name == 'cifar10':
        dataset = CIFAR10(
            root='./data', 
            train=train, 
            download=True,
            transform=T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        num_classes = 10
    elif name == 'cifar100':
        dataset = CIFAR100(
            root='./data', 
            train=train, 
            download=True,
            transform=T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset, num_classes