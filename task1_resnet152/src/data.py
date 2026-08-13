import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def get_default_data_dir():
    # Resolve absolute path to the main repo /data folder from src/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_cifar10_loaders(data_dir=None, batch_size=256, num_workers=2):
    """
    CIFAR-10 loader normalized with ImageNet mean & std using absolute repo data path.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_val)
    except Exception:
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))

    return train_loader, val_loader

def get_secondary_loaders(data_dir=None, batch_size=256, num_workers=2):
    """
    Secondary dataset loader (STL-10) using absolute repo data path.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = torchvision.datasets.STL10(root=data_dir, split='train', download=False, transform=transform)
        test_dataset = torchvision.datasets.STL10(root=data_dir, split='test', download=False, transform=transform)
    except Exception:
        train_dataset = torchvision.datasets.STL10(root=data_dir, split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.STL10(root=data_dir, split='test', download=True, transform=transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader
