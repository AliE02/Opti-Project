import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_loaders(data_dir, batch_size, augment, img_size, valid_size=0.1):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_transform = (
        test_transform
        if not augment
        else transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    )

    # load the dataset
    train_val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)

    valid_size = int(0.1 * len(train_val_dataset))
    train_size = len(train_val_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, valid_size]
    )

    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = test_transform

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print(
        f"Train dataset: {len(train_loader.dataset)} batch size {train_loader.batch_size} of dim {train_loader.dataset[0][0].shape}\nValidation dataset: {len(val_loader.dataset)} batch size {val_loader.batch_size} of dim {val_loader.dataset[0][0].shape}\nTest dataset: {len(test_loader.dataset)} batch size {test_loader.batch_size} of dim {test_loader.dataset[0][0].shape}"
    )

    return train_loader, val_loader, test_loader
