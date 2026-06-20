#!/usr/bin/env python3
"""
scripts/convert_kitti_to_yolo.py
─────────────────────────────────
Converts KITTI's 2D OBJECT DETECTION dataset (data_object_image_2 +
data_object_label_2 — NOT the tracking subset) into YOLO training format,
for fine-tuning a detector on KITTI's actual domain (camera angle, vehicle
styles, scene density) instead of relying on generic COCO pretraining.

Why the object-detection set, not the tracking set
────────────────────────────────────────────────────
KITTI's tracking sequences are a handful of continuous video clips — every
frame in a sequence is highly correlated with its neighbors, so 1000 tracking
frames carry far less training signal than 1000 independent images. The
object-detection set (~7481 labeled training images) is specifically curated
for detector training: diverse scenes, no frame-to-frame redundancy.

Class mapping
─────────────
KITTI has 8 raw classes. We collapse them to the 3 we evaluate on:
    Car, Van, Truck          -> Car         (class 0)
    Pedestrian, Person_sitting -> Pedestrian (class 1)
    Cyclist                  -> Cyclist     (class 2)
    Tram, Misc, DontCare      -> dropped entirely (not written to YOLO labels)

This mirrors the same merge logic used in the evaluator's ignore-region
handling, so the fine-tuned model's class semantics match exactly what
src/detector.py expects downstream (no further remapping needed at inference
time once you pass finetuned=True to KITTIDetector).

Output structure
─────────────────
yolo_dataset/
├── images/train/*.png  (symlinked, not copied — saves disk space)
├── images/val/*.png
├── labels/train/*.txt  (YOLO format: class cx cy w h, normalized 0-1)
├── labels/val/*.txt
└── dataset.yaml

Usage
─────
python scripts/convert_kitti_to_yolo.py \
    --kitti-images data/kitti_object/training/image_2 \
    --kitti-labels data/kitti_object/training/label_2 \
    --output data/yolo_dataset \
    --val-split 0.1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from rich.console import Console
from rich.progress import track

console = Console()

# ── Class mapping ──────────────────────────────────────────────────────────
# Maps raw KITTI class name -> (yolo_class_id, yolo_class_name)
# Classes not in this dict (Tram, Misc, DontCare) are dropped.
KITTI_TO_YOLO_CLASS: Dict[str, Tuple[int, str]] = {
    "Car":            (0, "Car"),
    "Van":            (0, "Car"),
    "Truck":          (0, "Car"),
    "Pedestrian":     (1, "Pedestrian"),
    "Person_sitting": (1, "Pedestrian"),
    "Cyclist":        (2, "Cyclist"),
}

YOLO_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]   # index == class id


def convert_kitti_box_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """
    Convert a KITTI [x1, y1, x2, y2] absolute-pixel box to YOLO's
    normalized [cx, cy, w, h] format (all in 0-1 range).
    """
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def parse_kitti_object_label(label_path: Path) -> List[dict]:
    """
    Parse a single KITTI object-detection label file (one file per image,
    NOT per-sequence like the tracking format — no frame index column).

    KITTI object detection label columns:
        type truncated occluded alpha
        bbox_left bbox_top bbox_right bbox_bottom
        h w l x y z rotation_y
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cls = parts[0]
            if cls not in KITTI_TO_YOLO_CLASS:
                continue
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            yolo_id, yolo_name = KITTI_TO_YOLO_CLASS[cls]
            boxes.append({
                "yolo_id": yolo_id,
                "yolo_name": yolo_name,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return boxes


def convert_dataset(
    kitti_images: Path,
    kitti_labels: Path,
    output_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(kitti_images.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {kitti_images}")

    console.print(f"Found {len(image_paths)} images")

    random.seed(seed)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    val_set = set(shuffled[:n_val])

    n_train_written, n_val_written, n_skipped_empty = 0, 0, 0

    for img_path in track(image_paths, description="Converting"):
        stem = img_path.stem
        label_path = kitti_labels / f"{stem}.txt"

        boxes = parse_kitti_object_label(label_path)
        if not boxes:
            n_skipped_empty += 1
            continue   # skip images with zero relevant objects after class filtering

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        split = "val" if img_path in val_set else "train"

        # Symlink image (saves disk — no need to duplicate KITTI's images)
        dst_img = output_dir / "images" / split / img_path.name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())

        # Write YOLO-format label
        dst_label = output_dir / "labels" / split / f"{stem}.txt"
        lines = []
        for box in boxes:
            cx, cy, w, h = convert_kitti_box_to_yolo(
                box["x1"], box["y1"], box["x2"], box["y2"], img_w, img_h
            )
            lines.append(f"{box['yolo_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dst_label.write_text("\n".join(lines))

        if split == "train":
            n_train_written += 1
        else:
            n_val_written += 1

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    yaml_content = f"""\
# Auto-generated by scripts/convert_kitti_to_yolo.py
path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: Car
  1: Pedestrian
  2: Cyclist
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)

    console.print(f"\n[bold green]Conversion complete[/]")
    console.print(f"  Train images: {n_train_written}")
    console.print(f"  Val images:   {n_val_written}")
    console.print(f"  Skipped (no relevant objects): {n_skipped_empty}")
    console.print(f"  Dataset config: {output_dir / 'dataset.yaml'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert KITTI object detection labels to YOLO format")
    p.add_argument("--kitti-images", type=Path, required=True,
                   help="Path to KITTI's training/image_2 directory")
    p.add_argument("--kitti-labels", type=Path, required=True,
                   help="Path to KITTI's training/label_2 directory")
    p.add_argument("--output", type=Path, default=Path("data/yolo_dataset"),
                   help="Output directory for YOLO-format dataset")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of images held out for validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    convert_dataset(
        kitti_images=args.kitti_images,
        kitti_labels=args.kitti_labels,
        output_dir=args.output,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
