"""
generate_dataset.py
--------------------
Builds a small, labelled road-scene image dataset on disk in the standard
ImageFolder layout expected by the rest of this project:

    data/
      train/<class>/*.png
      val/<class>/*.png
      test/<class>/*.png

WHY A GENERATED DATASET?
This project was built in an offline sandbox with no route to public dataset
hosts (Kaggle, Open Images, etc.). Rather than fake the pipeline, this script
procedurally draws simple road-scene "sprites" (car, bus, motorcycle, bicycle,
pedestrian) over randomised sky/road backgrounds, with randomised colour,
scale, position and jitter, so the images are genuinely learnable but not
trivial. This lets the ENTIRE pipeline (loading -> preprocessing -> transfer
learning -> evaluation -> demo) run end-to-end anywhere, in minutes, with no
downloads.

SWAPPING IN A REAL DATASET
This is a placeholder for real photographs. To use a real dataset instead,
just point data/train, data/val, data/test at folders of real .jpg/.png files
organised the same way (one sub-folder per class). No other code changes are
needed — dataset.py uses torchvision.datasets.ImageFolder, which reads this
exact layout regardless of whether the images are drawn or real.
"""

import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["car", "bus", "motorcycle", "bicycle", "pedestrian"]
IMG_SIZE = 96  # small on purpose: fast CPU training, upscaled by the model's own transforms if needed

# how many images per class per split
SPLIT_COUNTS = {"train": 150, "val": 25, "test": 25}

RNG_SEED = 42


def _sky_and_road(draw, w, h, rng):
    """Paint a randomised sky-gradient + grey road band as background."""
    sky_top = (rng.randint(120, 200), rng.randint(160, 220), rng.randint(200, 255))
    sky_bottom = (rng.randint(200, 240), rng.randint(210, 240), rng.randint(220, 250))
    for y in range(h):
        t = y / h
        r = int(sky_top[0] * (1 - t) + sky_bottom[0] * t)
        g = int(sky_top[1] * (1 - t) + sky_bottom[1] * t)
        b = int(sky_top[2] * (1 - t) + sky_bottom[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    road_h = rng.randint(int(h * 0.28), int(h * 0.4))
    road_grey = rng.randint(60, 110)
    draw.rectangle(
        [0, h - road_h, w, h],
        fill=(road_grey, road_grey, road_grey + rng.randint(-5, 5)),
    )
    # lane marking
    if rng.random() < 0.7:
        dash_y = h - road_h // 2
        x = 0
        while x < w:
            draw.line([(x, dash_y), (x + 8, dash_y)], fill=(230, 230, 200), width=2)
            x += 18
    return h - road_h  # y coordinate of the road's top edge


def _rand_color(rng, bright=True):
    lo, hi = (90, 230) if bright else (20, 120)
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def _draw_car(draw, cx, base_y, rng):
    body_w = rng.randint(34, 46)
    body_h = rng.randint(14, 20)
    color = _rand_color(rng)
    x0, y0 = cx - body_w // 2, base_y - body_h
    x1, y1 = cx + body_w // 2, base_y
    draw.rounded_rectangle([x0, y0, x1, y1], radius=4, fill=color, outline=(20, 20, 20))
    cabin_w = int(body_w * 0.55)
    cabin_h = int(body_h * 0.7)
    draw.rounded_rectangle(
        [cx - cabin_w // 2, y0 - cabin_h + 4, cx + cabin_w // 2, y0 + 4],
        radius=3, fill=color, outline=(20, 20, 20),
    )
    wheel_r = 4
    for wx in (x0 + 8, x1 - 8):
        draw.ellipse([wx - wheel_r, y1 - wheel_r, wx + wheel_r, y1 + wheel_r], fill=(15, 15, 15))


def _draw_bus(draw, cx, base_y, rng):
    body_w = rng.randint(58, 74)
    body_h = rng.randint(26, 34)
    color = _rand_color(rng)
    x0, y0 = cx - body_w // 2, base_y - body_h
    x1, y1 = cx + body_w // 2, base_y
    draw.rounded_rectangle([x0, y0, x1, y1], radius=3, fill=color, outline=(20, 20, 20))
    # windows row
    n_win = rng.randint(4, 6)
    win_w = (body_w - 10) / n_win
    for i in range(n_win):
        wx0 = x0 + 5 + i * win_w
        draw.rectangle([wx0, y0 + 5, wx0 + win_w * 0.7, y0 + body_h * 0.45],
                        fill=(210, 230, 240))
    wheel_r = 5
    for wx in (x0 + 10, x1 - 10):
        draw.ellipse([wx - wheel_r, y1 - wheel_r, wx + wheel_r, y1 + wheel_r], fill=(15, 15, 15))


def _draw_motorcycle(draw, cx, base_y, rng):
    # Deliberately distinct from the bicycle sprite: filled body mass + thick
    # solid wheels, vs. the bicycle's thin wireframe + hollow wheels.
    color = _rand_color(rng, bright=False)
    wheel_r = rng.randint(7, 9)
    gap = wheel_r * 2 + rng.randint(6, 10)
    wx1, wx2 = cx - gap // 2, cx + gap // 2
    for wx in (wx1, wx2):
        draw.ellipse([wx - wheel_r, base_y - wheel_r, wx + wheel_r, base_y + wheel_r],
                     fill=(15, 15, 15))
        draw.ellipse([wx - 2, base_y - 2, wx + 2, base_y + 2], fill=(90, 90, 90))
    # solid body/fuel-tank block sitting on the wheels
    body_w = gap + wheel_r
    body_h = rng.randint(9, 13)
    body_y1 = base_y - wheel_r
    body_y0 = body_y1 - body_h
    draw.rounded_rectangle([cx - body_w // 2, body_y0, cx + body_w // 2, body_y1],
                            radius=3, fill=color, outline=(15, 15, 15))
    # seat hump
    draw.ellipse([cx - 4, body_y0 - 4, cx + 4, body_y0 + 3], fill=color)
    # handlebar rising from the front wheel
    draw.line([(wx2, body_y0), (wx2 + 4, body_y0 - 10)], fill=(15, 15, 15), width=3)


def _draw_bicycle(draw, cx, base_y, rng):
    wheel_r = rng.randint(9, 12)
    gap = wheel_r * 2 + rng.randint(4, 8)
    wx1, wx2 = cx - gap // 2, cx + gap // 2
    for wx in (wx1, wx2):
        draw.ellipse([wx - wheel_r, base_y - wheel_r, wx + wheel_r, base_y + wheel_r],
                     outline=(20, 20, 20), width=2)
    frame_color = _rand_color(rng, bright=False)
    top_y = base_y - wheel_r - rng.randint(6, 10)
    draw.line([(wx1, base_y - wheel_r), (cx, top_y)], fill=frame_color, width=2)
    draw.line([(cx, top_y), (wx2, base_y - wheel_r)], fill=frame_color, width=2)
    draw.line([(wx1, base_y), (cx, top_y)], fill=frame_color, width=2)
    draw.line([(cx, top_y), (wx2, base_y)], fill=frame_color, width=2)
    draw.line([(wx2, base_y - wheel_r), (wx2 + 3, top_y - 6)], fill=(15, 15, 15), width=2)


def _draw_pedestrian(draw, cx, base_y, rng):
    height = rng.randint(30, 40)
    color = _rand_color(rng, bright=False)
    head_r = 4
    torso_h = int(height * 0.45)
    leg_h = height - torso_h - head_r * 2
    head_cy = base_y - height + head_r
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 fill=(230, 200, 170), outline=(20, 20, 20))
    torso_top = head_cy + head_r
    draw.line([(cx, torso_top), (cx, torso_top + torso_h)], fill=color, width=5)
    # arms
    draw.line([(cx, torso_top + 4), (cx - 6, torso_top + torso_h * 0.6)], fill=color, width=2)
    draw.line([(cx, torso_top + 4), (cx + 6, torso_top + torso_h * 0.6)], fill=color, width=2)
    leg_top = torso_top + torso_h
    stride = rng.randint(2, 6)
    draw.line([(cx, leg_top), (cx - stride, leg_top + leg_h)], fill=(30, 30, 60), width=3)
    draw.line([(cx, leg_top), (cx + stride, leg_top + leg_h)], fill=(30, 30, 60), width=3)


_DRAW_FNS = {
    "car": _draw_car,
    "bus": _draw_bus,
    "motorcycle": _draw_motorcycle,
    "bicycle": _draw_bicycle,
    "pedestrian": _draw_pedestrian,
}


def make_image(cls, rng):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)
    road_top = _sky_and_road(draw, IMG_SIZE, IMG_SIZE, rng)
    base_y = rng.randint(road_top + 6, IMG_SIZE - 6)
    cx = rng.randint(int(IMG_SIZE * 0.3), int(IMG_SIZE * 0.7))
    _DRAW_FNS[cls](draw, cx, base_y, rng)
    # mild jitter/noise + blur for realism, avoids the CNN memorising pixel-perfect edges
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.6)))
    return img


def build_dataset(root="data", seed=RNG_SEED, overwrite=True):
    root = Path(root)
    if overwrite and root.exists():
        shutil.rmtree(root)
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0}
    for split, n_per_class in SPLIT_COUNTS.items():
        for cls in CLASSES:
            out_dir = root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = make_image(cls, rng)
                img.save(out_dir / f"{cls}_{i:04d}.png")
                counts[split] += 1
    return counts


if __name__ == "__main__":
    counts = build_dataset()
    print("Dataset generated:")
    for split, n in counts.items():
        print(f"  {split}: {n} images across {len(CLASSES)} classes")
