"""
stress_test.py
---------------
The clean synthetic test set (evaluate.py) is solved perfectly by the model,
which leaves nothing real to put in an "error analysis" — and the project
brief explicitly asks for one. This script builds a deliberately HARDER,
corrupted variant of the test images (heavier noise, partial occlusion,
low-light darkening, extreme scale) and re-evaluates the same trained model
on it, so the report can discuss genuine failure modes, not a fabricated one.

This mirrors a standard real-world practice: stress-testing a vision model
against distribution shift (weather, occlusion, lighting) beyond the clean
validation set.

Run:  python3 stress_test.py   (after train.py has produced outputs/best_model.pt)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import confusion_matrix, classification_report

from generate_dataset import CLASSES, IMG_SIZE, _sky_and_road, _DRAW_FNS
from dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import build_model

OUT_DIR = Path("outputs")
STRESS_ROOT = Path("data_stress")
N_PER_CLASS = 20
SEED = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_hard_image(cls, rng):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)
    road_top = _sky_and_road(draw, IMG_SIZE, IMG_SIZE, rng)
    base_y = rng.randint(road_top + 6, IMG_SIZE - 6)
    # extreme scale: push objects smaller/further-away or larger/closer than training range
    cx = rng.randint(int(IMG_SIZE * 0.15), int(IMG_SIZE * 0.85))
    _DRAW_FNS[cls](draw, cx, base_y, rng)

    # partial occlusion: opaque patch over part of the object
    if rng.random() < 0.5:
        ow, oh = rng.randint(10, 22), rng.randint(10, 22)
        ox = cx + rng.randint(-15, 15)
        oy = base_y - rng.randint(5, 25)
        patch_color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.rectangle([ox, oy, ox + ow, oy + oh], fill=patch_color)

    # low-light darkening
    if rng.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        arr *= rng.uniform(0.35, 0.6)
        img = Image.fromarray(arr.astype(np.uint8))

    # heavy blur (motion / defocus)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 1.8)))

    # additive pixel noise
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 22, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img


def build_stress_set(root=STRESS_ROOT, n_per_class=N_PER_CLASS, seed=SEED):
    import shutil
    if root.exists():
        shutil.rmtree(root)
    rng = random.Random(seed)
    for cls in CLASSES:
        d = root / "test" / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = make_hard_image(cls, rng)
            img.save(d / f"{cls}_{i:03d}.png")


def unnormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def main():
    build_stress_set()
    print(f"Stress-test images written to {STRESS_ROOT}/test/<class>/")

    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    model, _ = build_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    _, eval_tf = get_transforms()
    from torchvision import datasets
    ds = datasets.ImageFolder(STRESS_ROOT / "test", transform=eval_tf)
    assert ds.classes == classes
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_images, all_probs = [], [], [], []
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1).cpu()

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            n += images.size(0)
            all_preds.append(preds); all_labels.append(labels)
            all_images.append(images); all_probs.append(probs.cpu())

    test_loss, test_acc = total_loss / n, correct / n
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_images = torch.cat(all_images)
    all_probs = torch.cat(all_probs).numpy()

    print(f"Stress-test loss: {test_loss:.4f}  accuracy: {test_acc:.4f}  (n={n})  "
          f"[clean test accuracy was 1.000]")
    report_txt = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report_txt)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Stress-test Confusion Matrix (acc={test_acc:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stress_confusion_matrix.png", dpi=130)
    plt.close(fig)

    wrong_idx = np.where(all_preds != all_labels)[0]
    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                errors.append((classes[i], classes[j], int(cm[i, j])))
    errors.sort(key=lambda x: -x[2])

    with open(OUT_DIR / "stress_error_analysis.txt", "w") as f:
        f.write(f"Stress-test accuracy: {test_acc:.4f} (clean test accuracy: 1.0000)\n")
        f.write(f"Corruptions applied: extreme scale/position, ~50% partial occlusion, "
                f"~50% low-light darkening, heavy Gaussian blur, additive pixel noise.\n\n")
        if errors:
            f.write("Confused class pairs (true -> predicted : count):\n")
            for true_c, pred_c, count in errors:
                f.write(f"  {true_c} -> {pred_c} : {count}\n")
        else:
            f.write("No misclassifications even under corruption.\n")
        f.write(f"\nTotal misclassified: {len(wrong_idx)} / {n}\n")
        for idx in wrong_idx:
            true_c, pred_c = classes[all_labels[idx]], classes[all_preds[idx]]
            conf = all_probs[idx][all_preds[idx]]
            f.write(f"  sample #{idx}: true={true_c} pred={pred_c} confidence={conf:.3f}\n")

    # sample grid: prioritise showing actual errors
    rng = np.random.default_rng(1)
    correct_idx = np.where(all_preds == all_labels)[0]
    show_idx = list(wrong_idx[:10])
    remaining = 10 - len(show_idx)
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
        axes[ax_i].imshow(img); axes[ax_i].axis("off")
        axes[ax_i].set_title(f"true: {true_c}\npred: {pred_c} ({conf:.2f})",
                              color="green" if ok else "red", fontsize=9)
    for ax_i in range(len(show_idx), len(axes)):
        axes[ax_i].axis("off")
    fig.suptitle("Stress-test predictions (green=correct, red=incorrect)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(OUT_DIR / "stress_sample_predictions.png", dpi=130)
    plt.close(fig)

    with open(OUT_DIR / "stress_test_summary.json", "w") as f:
        json.dump({"clean_test_acc": 1.0, "stress_test_acc": test_acc,
                    "stress_test_loss": test_loss, "n": n,
                    "confused_pairs": errors}, f, indent=2)

    print(f"Saved: {OUT_DIR/'stress_confusion_matrix.png'}, "
          f"{OUT_DIR/'stress_sample_predictions.png'}, {OUT_DIR/'stress_error_analysis.txt'}")


if __name__ == "__main__":
    main()
