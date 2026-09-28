"""
dataset.py
----------
Preprocessing pipeline and DataLoaders for the road-scene classifier.

Uses torchvision.datasets.ImageFolder, which expects:
    data/<split>/<class_name>/<image files>

This is a completely standard layout — pointing DATA_ROOT at a folder of real
photographs organised the same way works with zero other code changes.
"""

from pathlib import Path

import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Resize target for the backbone. Kept modest (96px) so the whole project
# trains in a few minutes on a CPU; raise to 224 for a real dataset on a GPU.
INPUT_SIZE = 96


def get_transforms(input_size=INPUT_SIZE):
    """Separate train/eval preprocessing pipelines.

    Train: light augmentation (flip, rotation, colour jitter) so the model
    doesn't just memorise exact pixel layouts.
    Eval (val/test/demo): deterministic resize + normalise only, so evaluation
    numbers reflect the model, not random augmentation.
    """
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_root="data", batch_size=32, input_size=INPUT_SIZE, num_workers=0):
    data_root = Path(data_root)
    train_tf, eval_tf = get_transforms(input_size)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    # ImageFolder assigns class indices alphabetically; keep this mapping
    # around, everything downstream (reports, confusion matrix) relies on it.
    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "train/val/test must have identical class folders"

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes
