# Road Scene Object Classification (Semester 3 — Deep Learning, Project 17)

Transfer-learning image classifier that identifies 5 common road-scene
objects: **car, bus, motorcycle, bicycle, pedestrian**.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Run the full pipeline, in order

```bash
python3 generate_dataset.py   # builds data/train, data/val, data/test
python3 train.py              # transfer learning: trains + saves outputs/best_model.pt
python3 evaluate.py           # test accuracy, confusion matrix, classification report
python3 stress_test.py        # harder corrupted set -> genuine error analysis
python3 demo.py [image.jpg]   # working demonstration on one image (random test image if omitted)
```

Everything is deterministic (fixed random seeds) and CPU-friendly — the
whole pipeline runs in a few minutes with no GPU.

## 3. Project layout

| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds the labelled image dataset (see note below) |
| `dataset.py` | Preprocessing pipeline + DataLoaders (`ImageFolder`-based) |
| `model.py` | Transfer-learning model: ResNet-18 (ImageNet-pretrained) + new head |
| `train.py` | Two-stage training: head-only, then fine-tune last block |
| `evaluate.py` | Test accuracy, classification report, confusion matrix, sample predictions |
| `stress_test.py` | Harder corrupted test set + confusion matrix, for real error analysis |
| `demo.py` | Loads the trained model and classifies one image (the "working demo") |
| `outputs/` | All generated artefacts: checkpoint, plots, reports |

## 4. About the dataset

This project was built in an offline environment with no route to public
dataset hosts. `generate_dataset.py` therefore **procedurally draws** simple,
labelled road-scene sprites (car/bus/motorcycle/bicycle/pedestrian) over
randomised sky/road backgrounds, with randomised colour, scale, position and
noise, in the exact `data/<split>/<class>/` folder layout `ImageFolder`
expects.

**To use a real photo dataset instead**, just replace the contents of
`data/train`, `data/val`, `data/test` with folders of real images organised
the same way (one sub-folder per class name). No other code changes are
required — `dataset.py` and everything downstream is dataset-agnostic.

## 5. About the "pretrained" weights fallback

`model.py` downloads standard ImageNet-pretrained ResNet-18 weights from
PyTorch's model hub. If that download is blocked (as it is in this sandbox —
see the printed warning when you run `train.py`), the code automatically
falls back to training the network from random initialisation instead of
silently producing a broken model. **On a machine with normal internet
access this fallback never triggers**, and you get genuine transfer learning
(frozen pretrained backbone + trained head, then fine-tuning) as the project
brief requires.
