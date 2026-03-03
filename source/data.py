# Dataset and dataloaders

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from config import DATA_DIR, SEED, NUM_WORKERS, PIN_MEMORY



def get_datasets():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    return trainset, testset



def get_split_indices(dataset_length, train_ratio=0.9):

    train_size = int(train_ratio * dataset_length)
    val_size = dataset_length - train_size

    # Ensure reproducible train/validation split for tuning with Optuna
    generator = torch.Generator().manual_seed(SEED)

    train_indices, val_indices = random_split(
        range(dataset_length),
        [train_size, val_size],
        generator=generator
    )

    return train_indices, val_indices


def get_dataloaders(batch_size):
    trainset, _ = get_datasets()

    train_indices, val_indices = get_split_indices(len(trainset))

    train_dataset = Subset(trainset, train_indices)
    val_dataset = Subset(trainset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size):
    _, testset = get_datasets()

    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


def get_dataset_metadata():

    trainset, testset = get_datasets()

    return {
        "num_train_samples": len(trainset),
        "num_test_samples": len(testset),
        "num_classes": 10,
        "input_shape": (3, 32, 32)
    }
# in train.py use : mlflow.log_params(get_dataset_metadata())
