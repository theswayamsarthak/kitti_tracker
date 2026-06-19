#!/usr/bin/env python3
"""
scripts/download_kitti.py
─────────────────────────
Helper to download and organize the KITTI Multi-Object Tracking dataset.

KITTI tracking data must be downloaded from:
  https://www.cvlibs.net/datasets/kitti/eval_tracking.php

Required files:
  - data_tracking_image_2.zip   (image sequences, ~1.5 GB)
  - data_tracking_label_2.zip   (labels, ~4 MB)
  - data_tracking_calib.zip     (calibration, ~1 MB)

This script:
  1. Provides download instructions (KITTI requires registration)
  2. Can extract and organize already-downloaded zip files
  3. Validates the resulting directory structure

Usage
─────
# Just show download instructions:
python scripts/download_kitti.py --info

# Extract zips you already downloaded to data/kitti:
python scripts/download_kitti.py \
    --images  /path/to/data_tracking_image_2.zip \
    --labels  /path/to/data_tracking_label_2.zip \
    --calib   /path/to/data_tracking_calib.zip \
    --output  data/kitti

# Validate existing KITTI directory:
python scripts/download_kitti.py --validate --kitti-root data/kitti
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

KITTI_URL    = "https://www.cvlibs.net/datasets/kitti/eval_tracking.php"
REQUIRED_ZIPS = {
    "images": "data_tracking_image_2.zip",
    "labels": "data_tracking_label_2.zip",
    "calib":  "data_tracking_calib.zip",
}


def print_instructions() -> None:
    console.print(Panel(
        "[bold]KITTI Tracking Dataset Download Instructions[/]\n\n"
        "1. Register (free) at: [link]https://www.cvlibs.net/[/link]\n"
        "2. Go to: [cyan]https://www.cvlibs.net/datasets/kitti/eval_tracking.php[/]\n"
        "3. Download these 3 files:\n"
        "   • [green]Left color images of tracking data set[/]   (~1.5 GB)\n"
        "   • [green]Tracking labels[/]                          (~4 MB)\n"
        "   • [green]Camera calibration[/]                       (~1 MB)\n\n"
        "4. Run this script with --images, --labels, --calib flags\n\n"
        "[dim]Alternative: Use Kaggle mirror (no registration needed):[/]\n"
        "  kaggle datasets download -d [cyan]klemenko/kitti-dataset[/]",
        title="Setup",
        expand=False,
    ))


def extract_zip(zip_path: Path, output_dir: Path, desc: str) -> None:
    console.print(f"Extracting [cyan]{desc}[/] from {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    console.print(f"  [green]Done[/]")


def organize_kitti(root: Path) -> None:
    """
    After extraction, KITTI zips usually unpack as:
      training/image_02/…  (or data_tracking_image_2/training/image_02/…)
    Ensure they live at root/training/…
    """
    # Handle nested extraction paths
    for nested in [
        root / "data_tracking_image_2",
        root / "data_tracking_label_2",
        root / "data_tracking_calib",
    ]:
        if nested.exists():
            for child in nested.iterdir():
                target = root / child.name
                if not target.exists():
                    shutil.move(str(child), str(target))
                    console.print(f"  Moved {child.name} → {root}/")
            nested.rmdir()


def validate(kitti_root: Path) -> bool:
    """Check the directory structure is correct."""
    expected = [
        kitti_root / "training" / "image_02",
        kitti_root / "training" / "label_02",
        kitti_root / "training" / "calib",
    ]
    ok = True
    for path in expected:
        if path.exists():
            n = len(list(path.iterdir()))
            console.print(f"  [green]✓[/] {path.relative_to(kitti_root.parent)}  ({n} items)")
        else:
            console.print(f"  [red]✗[/] Missing: {path.relative_to(kitti_root.parent)}")
            ok = False

    if ok:
        # Count sequences
        seqs = sorted(kitti_root.glob("training/image_02/*"))
        console.print(f"\n  [bold green]Dataset OK — {len(seqs)} sequences found[/]")
    else:
        console.print("\n  [bold red]Dataset validation FAILED[/]")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="KITTI Tracking dataset setup")
    p.add_argument("--info",       action="store_true",
                   help="Print download instructions")
    p.add_argument("--images",     type=Path, default=None,
                   help="Path to data_tracking_image_2.zip")
    p.add_argument("--labels",     type=Path, default=None,
                   help="Path to data_tracking_label_2.zip")
    p.add_argument("--calib",      type=Path, default=None,
                   help="Path to data_tracking_calib.zip")
    p.add_argument("--output",     type=Path, default=Path("data/kitti"),
                   help="Output directory for KITTI data")
    p.add_argument("--validate",   action="store_true",
                   help="Validate existing KITTI directory structure")
    p.add_argument("--kitti-root", type=Path, default=Path("data/kitti"),
                   help="KITTI root for --validate")
    args = p.parse_args()

    if args.info or (not args.images and not args.validate):
        print_instructions()
        return

    if args.validate:
        console.print(f"\n[bold]Validating {args.kitti_root} …[/]")
        validate(args.kitti_root)
        return

    # Extract
    args.output.mkdir(parents=True, exist_ok=True)
    if args.images:
        extract_zip(args.images, args.output, "images")
    if args.labels:
        extract_zip(args.labels, args.output, "labels")
    if args.calib:
        extract_zip(args.calib,  args.output, "calibration")

    organize_kitti(args.output)

    console.print(f"\n[bold]Validating …[/]")
    validate(args.output)


if __name__ == "__main__":
    main()
