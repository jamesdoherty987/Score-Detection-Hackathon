"""
Train a YOLOv8 football detector.

  python training/train_football.py

Labeled images: dataset/football/yolo/train/images + labels
Optional test:  dataset/football/yolo/test/images + labels
New stills:     dataset/football/raw/  (label them, then copy into train/)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "dataset" / "football" / "yolo" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
MODELS_DIR = PROJECT_ROOT / "models"

BASE_MODEL = "yolov8n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH = 8


def main() -> int:
    if not DATA_YAML.exists():
        raise SystemExit(f"Missing dataset config: {DATA_YAML}")

    train_img = DATA_YAML.parent / "train" / "images"
    n = len(list(train_img.glob("*.*"))) if train_img.is_dir() else 0
    if n == 0:
        raise SystemExit(f"No training images in {train_img}")

    # Ultralytics resolves train/val relative to `path` — use an absolute folder.
    yaml_text = DATA_YAML.read_text()
    lines = []
    for line in yaml_text.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {DATA_YAML.parent}")
        else:
            lines.append(line)
    DATA_YAML.write_text("\n".join(lines) + "\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data: {DATA_YAML}  ({n} train images)")
    print(f"Base model: {BASE_MODEL}")
    print(f"Epochs: {EPOCHS}, imgsz: {IMAGE_SIZE}, batch: {BATCH}")

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        project=str(RUNS_DIR),
        name="football",
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    if best.exists():
        dest = MODELS_DIR / "football_yolov8n.pt"
        shutil.copy2(best, dest)
        print(f"Copied best weights -> {dest}")
    elif last.exists():
        dest = MODELS_DIR / "football_yolov8n.pt"
        shutil.copy2(last, dest)
        print(f"Copied last weights -> {dest}")
    else:
        print("Training finished but no weights file found.")

    print("Done. Use models/football_yolov8n.pt with score_live.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
