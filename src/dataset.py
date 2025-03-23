import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_transformations(dataset_name):
    """
    Returns appropriate data transformations for the given dataset
    """
    if dataset_name == "CIFAR-10":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    
    elif dataset_name == "Fashion-MNIST":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_transform, test_transform

def get_dataset(name, download=True):
    """
    Returns train and test datasets
    """
    root = './data'
    train_transform, test_transform = get_data_transformations(name)
    
    if name == "CIFAR-10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    elif name == "Fashion-MNIST":
        train_dataset = datasets.FashionMNIST(root=root, train=True, download=download, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=download, transform=test_transform)
        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return train_dataset, test_dataset, classes

def get_data_loaders(train_dataset, test_dataset, batch_size=64, val_split=0.1):
    """
    Creates data loaders for training, validation and testing
    """
    # Create validation split
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader