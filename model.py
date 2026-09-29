"""
model.py
--------
Transfer-learning model: a torchvision ResNet-18 backbone (pretrained on
ImageNet) with its final fully-connected layer replaced by a new head sized
for our 5 road-scene classes.

Standard transfer-learning recipe implemented here:
  1. Load ResNet-18 with ImageNet-pretrained weights.
  2. Freeze every convolutional layer (they already extract generic edge /
     texture / shape features that transfer well to new images).
  3. Replace the final classifier layer with a fresh Linear(., num_classes)
     and train ONLY that layer first (fast, and prevents the random new head
     from wrecking the pretrained features).
  4. Optionally "fine-tune": unfreeze the last conv block and continue
     training the whole thing at a low learning rate for a few more epochs,
     to adapt the pretrained features to this specific task.

OFFLINE FALLBACK
Pretrained ImageNet weights are downloaded from PyTorch's model hub the first
time this code runs. If that download is unreachable (e.g. an offline
sandbox, an air-gapped machine), we catch the failure, fall back to random
initialisation, and automatically switch off backbone freezing — freezing a
backbone that never saw real images would just prevent it from learning
anything. A clear warning is printed either way so it's obvious which mode
ran. On a normal internet connection this fallback never triggers.
"""

import warnings

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True):
    """Returns (model, used_pretrained_weights: bool).

    Builds a ResNet-18 for transfer learning: the ImageNet-pretrained
    backbone is (optionally) frozen and the final classification layer is
    replaced with a new one sized for `num_classes`.
    """
    # Tracks whether we actually got pretrained weights, so the caller can
    # tell a real transfer-learning run from a fallback sanity-check run.
    used_pretrained = True
    try:
        # Downloads the ImageNet weights on first use (cached afterwards).
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:  # network unreachable, hub down, etc.
        warnings.warn(
            f"Could not download ImageNet-pretrained weights ({e!r}). "
            "Falling back to random initialisation and disabling backbone "
            "freezing. Results in this mode are a pipeline sanity-check, "
            "NOT a measure of transfer-learning accuracy — re-run with "
            "internet access for the real result."
        )
        # Random init: same architecture, no learned features.
        backbone = models.resnet18(weights=None)
        used_pretrained = False
        # Freezing random weights would make the model untrainable in practice
        # (only the head would learn on top of noise), so force full training.
        freeze_backbone = False

    if freeze_backbone:
        # Feature extraction mode: keep pretrained weights fixed so only the
        # new head is updated. Faster and less prone to overfitting on small
        # datasets.
        for param in backbone.parameters():
            param.requires_grad = False

    # ResNet-18's original head maps 512 features -> 1000 ImageNet classes.
    # Swap it for a layer that outputs our own number of classes.
    in_features = backbone.fc.in_features
    # Newly created layers have requires_grad=True by default, so this head
    # is trainable even if the rest of the backbone was frozen above.
    backbone.fc = nn.Linear(in_features, num_classes)  # new head is always trainable

    return backbone, used_pretrained


def unfreeze_last_block(model):
    """Unfreeze layer4 (the last residual block) + the head for fine-tuning.

    Typical two-stage workflow: first train just the head with the backbone
    frozen, then call this and continue training with a lower learning rate
    so the deepest (most task-specific) features can adapt to your data.
    """
    for name, param in model.named_parameters():
        # Parameter names look like "layer4.1.conv2.weight" or "fc.bias",
        # so a prefix check selects the last block and the classifier.
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
    return model


def trainable_parameters(model):
    """Return only the parameters the optimizer should update.

    Pass this to the optimizer (e.g. torch.optim.Adam) so frozen weights
    aren't tracked. If you unfreeze more layers later, rebuild the optimizer.
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model):
    """Return (total, trainable) parameter counts, handy for sanity-checking
    that freezing/unfreezing did what you expected."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
