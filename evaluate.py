"""
evaluate.py
-----------
Loads the best checkpoint from train.py and reports:
  - overall test accuracy
  - per-class precision/recall/F1 (classification report)
  - confusion matrix (plot + saved array)
  - a grid of sample predictions (correct + incorrect) for the report
  - explicit error analysis: which classes get confused with which, and why

Run:  python3 evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import build_model

OUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unnormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def main():
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model, _ = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    _, _, test_loader, ds_classes = get_dataloaders(data_root="data", batch_size=32)
    assert ds_classes == classes, "class order mismatch between checkpoint and dataset"

    all_preds, all_labels, all_images, all_probs = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images_d, labels_d = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images_d)
            loss = criterion(outputs, labels_d)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels_d).sum().item()
            n += images.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_images.append(images.cpu())
            all_probs.append(probs.cpu())

    test_loss = total_loss / n
    test_acc = correct / n
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_images = torch.cat(all_images)
    all_probs = torch.cat(all_probs).numpy()

    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}  (n={n})")

    # ---- classification report ----
    report_txt = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report_txt)
    report_dict = classification_report(all_labels, all_preds, target_names=classes,
                                         digits=3, output_dict=True)
    with open(OUT_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n")
        f.write(report_txt)

    # ---- confusion matrix ----
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (test acc={test_acc:.3f})")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    np.save(OUT_DIR / "confusion_matrix.npy", cm)

    # ---- error analysis: which pairs get confused, and worked examples ----
    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                errors.append((classes[i], classes[j], int(cm[i, j])))
    errors.sort(key=lambda x: -x[2])

    with open(OUT_DIR / "error_analysis.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        if errors:
            f.write("Confused class pairs (true -> predicted : count):\n")
            for true_c, pred_c, count in errors:
                f.write(f"  {true_c} -> {pred_c} : {count}\n")
        else:
            f.write("No misclassifications on the test set.\n")
        wrong_idx = np.where(all_preds != all_labels)[0]
        f.write(f"\nTotal misclassified: {len(wrong_idx)} / {n}\n")
        for idx in wrong_idx:
            true_c, pred_c = classes[all_labels[idx]], classes[all_preds[idx]]
            conf = all_probs[idx][all_preds[idx]]
            f.write(f"  sample #{idx}: true={true_c} pred={pred_c} confidence={conf:.3f}\n")
    print(f"Misclassified on test set: {len(wrong_idx)} / {n}")

    # ---- sample prediction grid (mix of correct + incorrect if any) ----
    rng = np.random.default_rng(0)
    correct_idx = np.where(all_preds == all_labels)[0]
    n_show = min(10, n)
    show_idx = list(wrong_idx[:5]) if len(wrong_idx) else []
    remaining = n_show - len(show_idx)
    if remaining > 0:
        pool = [i for i in correct_idx if i not in show_idx]
        show_idx += list(rng.choice(pool, size=min(remaining, len(pool)), replace=False))

    cols = 5
    rows = int(np.ceil(len(show_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.array(axes).reshape(-1)
    for ax_i, idx in enumerate(show_idx):
        img = unnormalize(all_images[idx]).permute(1, 2, 0).numpy()
        true_c, pred_c = classes[all_labels[idx]], classes[all_preds[idx]]
        conf = all_probs[idx][all_preds[idx]]
        ok = true_c == pred_c
        axes[ax_i].imshow(img)
        axes[ax_i].axis("off")
        axes[ax_i].set_title(f"true: {true_c}\npred: {pred_c} ({conf:.2f})",
                              color="green" if ok else "red", fontsize=9)
    for ax_i in range(len(show_idx), len(axes)):
        axes[ax_i].axis("off")
    fig.suptitle("Sample predictions (green=correct, red=incorrect)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(OUT_DIR / "sample_predictions.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved: {OUT_DIR/'confusion_matrix.png'}, {OUT_DIR/'sample_predictions.png'}, "
          f"{OUT_DIR/'classification_report.txt'}, {OUT_DIR/'error_analysis.txt'}")


if __name__ == "__main__":
    main()
