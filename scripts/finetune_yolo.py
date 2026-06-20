#!/usr/bin/env python3
"""
scripts/finetune_yolo.py
─────────────────────────
Fine-tune a YOLO model on KITTI's own domain (camera angle, vehicle styles,
scene density) instead of relying on generic COCO pretraining.

Run scripts/convert_kitti_to_yolo.py FIRST to produce the dataset.yaml this
script expects.

Usage
─────
python scripts/finetune_yolo.py \
    --data data/yolo_dataset/dataset.yaml \
    --model yolo26m.pt \
    --epochs 50 \
    --imgsz 1280 \
    --batch 16 \
    --device 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich.console import Console
from ultralytics import YOLO

console = Console()


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLO on KITTI")
    p.add_argument("--data", type=Path, required=True,
                   help="Path to dataset.yaml from convert_kitti_to_yolo.py")
    p.add_argument("--model", default="yolo26m.pt",
                   help="Base pretrained checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0",
                   help="'0' for first GPU, 'cpu' for CPU, '0,1' for multi-GPU")
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (epochs with no improvement)")
    p.add_argument("--project", type=Path, default=Path("runs/finetune"))
    p.add_argument("--name", default="kitti_yolo")
    p.add_argument("--output-checkpoint", type=Path, default=Path("checkpoints/kitti_finetuned.pt"),
                   help="Where to copy the best checkpoint after training")
    args = p.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found — run scripts/convert_kitti_to_yolo.py first"
        )

    console.print(f"[bold]Fine-tuning {args.model} on {args.data}[/]")
    console.print(f"  Epochs: {args.epochs}  ImgSize: {args.imgsz}  Batch: {args.batch}  Device: {args.device}")

    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        # KITTI-relevant augmentation tweaks — vehicles in dashcam footage
        # don't appear upside-down or heavily rotated, so we tone down
        # augmentations that don't reflect this domain's real variation.
        degrees=0.0,        # no rotation augmentation
        flipud=0.0,         # no vertical flip — cars are never upside down
        fliplr=0.5,         # horizontal flip is fine (mirrored street scenes)
        mosaic=1.0,         # keep mosaic — helps with small/distant objects
    )

    # Locate best checkpoint
    best_ckpt = args.project / args.name / "weights" / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {best_ckpt}")

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_ckpt, args.output_checkpoint)

    console.print(f"\n[bold green]Training complete[/]")
    console.print(f"  Best checkpoint: {best_ckpt}")
    console.print(f"  Copied to: {args.output_checkpoint}")
    console.print(f"\nUse this checkpoint in the pipeline with:")
    console.print(f"  KITTIDetector(model_path='{args.output_checkpoint}', finetuned=True)")


if __name__ == "__main__":
    main()S
