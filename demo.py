"""
demo.py
-------
The "working demonstration" deliverable: loads the trained model and
classifies a single road-scene image, printing the predicted class and the
model's confidence for every class.

Usage:
    python3 demo.py path/to/image.jpg
    python3 demo.py                       # no path given -> picks a random
                                           # test image so the demo always runs

Also saves outputs/demo_prediction.png — the image with its predicted label,
for the report / screen-recording of the "working demonstration".
"""

import random
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import get_transforms
from model import build_model

OUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    model, _ = build_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, classes


def predict(model, classes, image_path):
    _, eval_tf = get_transforms()
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    order = torch.argsort(probs, descending=True)
    ranked = [(classes[i], probs[i].item()) for i in order]
    return img, ranked


def pick_random_test_image():
    test_dir = Path("data/test")
    all_imgs = list(test_dir.glob("*/*.png"))
    if not all_imgs:
        raise SystemExit("No test images found. Run generate_dataset.py first.")
    return random.choice(all_imgs)


def main():
    model, classes = load_model()

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = pick_random_test_image()
        print(f"No image path given — using a random test image: {image_path}")

    img, ranked = predict(model, classes, image_path)

    print(f"\nImage: {image_path}")
    print("Predictions (most to least likely):")
    for cls, p in ranked:
        bar = "#" * int(p * 30)
        print(f"  {cls:>12s}  {p:6.2%}  {bar}")

    top_class, top_p = ranked[0]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Prediction: {top_class} ({top_p:.1%})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demo_prediction.png", dpi=130)
    print(f"\nSaved annotated result -> {OUT_DIR / 'demo_prediction.png'}")


if __name__ == "__main__":
    main()
