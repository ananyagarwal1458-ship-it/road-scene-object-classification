"""
train.py
--------
Two-stage transfer-learning training loop:

  Stage A (head-only):  backbone frozen, train only the new classifier head.
  Stage B (fine-tune):  unfreeze the last conv block, continue training the
                         whole thing at a lower learning rate.

If pretrained weights could not be downloaded (see model.py), Stage A trains
the entire randomly-initialised network instead (freezing is skipped
automatically), and epoch counts are increased since there's no pretrained
head-start.

Run:  python3 train.py
Saves:
  outputs/best_model.pt          - best checkpoint (by val accuracy)
  outputs/training_curves.png    - loss & accuracy curves
  outputs/training_log.json      - per-epoch metrics
  outputs/class_names.json       - class index -> name mapping
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import build_model, unfreeze_last_block, trainable_parameters, count_params

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over `loader`. If optimizer is given, trains; else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)

    return total_loss / n, correct / n


def train_stage(model, train_loader, val_loader, criterion, lr, n_epochs, stage_name, history, best_state):
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        dt = time.time() - t0

        history["epoch"].append(len(history["epoch"]) + 1)
        history["stage"].append(stage_name)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{stage_name}] epoch {epoch}/{n_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({dt:.1f}s)")

        if val_acc > best_state["val_acc"]:
            best_state["val_acc"] = val_acc
            best_state["state_dict"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_state["epoch"] = history["epoch"][-1]


def main():
    torch.manual_seed(42)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_root="data", batch_size=32)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}  "
          f"Test batches: {len(test_loader)}")

    model, used_pretrained = build_model(num_classes=num_classes, freeze_backbone=True)
    model.to(DEVICE)

    total_p, trainable_p = count_params(model)
    print(f"Pretrained weights used: {used_pretrained}")
    print(f"Total params: {total_p:,} | Trainable (stage A): {trainable_p:,}")

    criterion = nn.CrossEntropyLoss()
    history = {"epoch": [], "stage": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = {"val_acc": -1.0, "state_dict": None, "epoch": None}

    if used_pretrained:
        # Stage A: head-only (backbone frozen)
        train_stage(model, train_loader, val_loader, criterion,
                    lr=1e-3, n_epochs=8, stage_name="A-head-only",
                    history=history, best_state=best_state)
        # Stage B: unfreeze last block, fine-tune at a lower LR
        unfreeze_last_block(model)
        total_p, trainable_p = count_params(model)
        print(f"Trainable (stage B, fine-tune): {trainable_p:,}")
        train_stage(model, train_loader, val_loader, criterion,
                    lr=1e-4, n_epochs=5, stage_name="B-fine-tune",
                    history=history, best_state=best_state)
    else:
        # No pretrained head-start available -> train everything from scratch,
        # for more epochs, as a pipeline sanity-check (see model.py warning).
        train_stage(model, train_loader, val_loader, criterion,
                    lr=1e-3, n_epochs=18, stage_name="A-from-scratch",
                    history=history, best_state=best_state)

    # restore best checkpoint by val accuracy before final save/eval
    model.load_state_dict(best_state["state_dict"])
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "used_pretrained": used_pretrained,
        "best_val_acc": best_state["val_acc"],
        "best_epoch": best_state["epoch"],
    }, OUT_DIR / "best_model.pt")

    with open(OUT_DIR / "class_names.json", "w") as f:
        json.dump(classes, f, indent=2)
    with open(OUT_DIR / "training_log.json", "w") as f:
        json.dump({"history": history, "used_pretrained": used_pretrained,
                    "best_val_acc": best_state["val_acc"], "best_epoch": best_state["epoch"]},
                   f, indent=2)

    # plot curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(f"Training curves (pretrained={used_pretrained})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=130)
    print(f"\nBest val_acc={best_state['val_acc']:.3f} at epoch {best_state['epoch']}")
    print(f"Saved checkpoint -> {OUT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
