
"""
Bayesian DL via SDEs

Helper function for data loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from PIL import Image

EPS = 1e-5  # define a small constant for numerical stability control

# data loading
def mnist(path='../data', batch_size=100, seed=0):
    # setting up the MNIST dataset
    transform=transforms.Compose([transforms.ToTensor(),])
    train_data = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(path, train=False, transform=transform)
    # split the training data to get validation data
    val_size = 6000
    train_size = len(train_data) - val_size
    torch.manual_seed(seed)
    train_data, val_data = random_split(train_data, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    input_shape = (1, 28, 28)
    n_class = 10
    return train_loader, val_loader, test_loader, input_shape, n_class

def cifar10(path='../data', batch_size=100, seed=0, image_size=None):
    aug_tf = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip()]
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    norm_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if image_size is not None:
        aug_tf = [transforms.RandomResizedCrop(image_size, antialias=True), transforms.RandomHorizontalFlip()]
    else:
        aug_tf = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip()]
    train_data = datasets.CIFAR10(root=path, train=True, download=True,
                              transform=transforms.Compose(aug_tf + norm_tf))
    # split the training data to get validation data
    val_size = 5000
    train_size = len(train_data) - val_size
    torch.manual_seed(seed)
    train_data_, val_data = random_split(train_data, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data_, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
    if image_size is not None:
        norm_tf = norm_tf + [transforms.Resize(image_size, antialias=True), transforms.CenterCrop(image_size)]
    test_data = datasets.CIFAR10(root=path, train=False, download=True,
                             transform=transforms.Compose(norm_tf))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    if image_size is None:
        input_shape = (3, 32, 32)
    else:
        input_shape = (3, image_size, image_size)
    n_class = 10
    return train_loader, val_loader, test_loader, input_shape, n_class

