#!/usr/bin/env python3
"""
run_tracker.py
──────────────
Main CLI for the KITTI Vehicle Tracker.

Usage examples
──────────────
# Run full pipeline on all training sequences:
python run_tracker.py

# Custom config:
python run_tracker.py --config configs/config.yaml

# Only specific sequences:
python run_tracker.py --sequences 0 1 2

# Skip video output (evaluation only, faster):
python run_tracker.py --no-video

# Use a custom YOLOv8 checkpoint:
python run_tracker.py --model yolov8l.pt

# Evaluate only (no video rendering):
python run_tracker.py --eval-only
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

# ── Make src importable when running from repo root ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import KITTITrackingDataset
from src.detector import KITTIDetector
from src.evaluator import MOTAEvaluator
from src.pipeline import run_all_sequences
from src.tracker import KITTITracker

console = Console()


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Pretty results table ──────────────────────────────────────────────────────

def _print_results(df: pd.DataFrame) -> None:
    table = Table(title="MOTA Results", show_header=True, header_style="bold cyan")
    cols  = [
        ("Seq",   "seq_name"),
        ("Frames","n_frames"),
        ("MOTA ↑","mota"),
        ("MOTP ↑","motp"),
        ("IDF1 ↑","idf1"),
        ("IDS ↓", "num_switches"),
        ("FP ↓",  "num_false_positives"),
        ("FN ↓",  "num_misses"),
        ("Rec ↑", "recall"),
        ("Prec ↑","precision"),
        ("FPS",   "fps"),
    ]

    present = set(df.columns)
    shown   = [(h, k) for h, k in cols if k in present]

    for header, _ in shown:
        table.add_column(header, justify="right")

    for _, row in df.iterrows():
        table.add_row(*[str(row.get(k, "—")) for _, k in shown])

    # Mean row
    numeric = [k for _, k in shown if k not in ("seq_name",)]
    means   = df[numeric].mean()
    mean_row= ["MEAN" if k == "seq_name" else
               f"{means[k]:.2f}" if k in means else "—"
               for _, k in shown]
    table.add_row(*mean_row, style="bold")

    console.print(table)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KITTI Vehicle Tracker — YOLO + ByteTrack + MOTA Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default="configs/config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--kitti-root", default=None,
                   help="Override KITTI root directory from config")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory from config")
    p.add_argument("--model",     default=None,
                   help="Override YOLO model path (e.g. yolov8l.pt)")
    p.add_argument("--sequences", nargs="+", type=int, default=None,
                   help="Specific sequence IDs to process (default: all)")
    p.add_argument("--device",    default=None,
                   help="Device override: cpu / cuda / cuda:0")
    p.add_argument("--no-video",  action="store_true",
                   help="Skip video rendering (evaluation only)")
    p.add_argument("--eval-only", action="store_true",
                   help="Same as --no-video")
    p.add_argument("--demo-seq",  type=int, default=0,
                   help="Sequence ID to name as tracking_demo.mp4")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    kitti_root = args.kitti_root or cfg["paths"]["kitti_root"]
    output_dir = args.output_dir or cfg["paths"]["output_dir"]
    model_path = args.model      or cfg["detection"]["model"]
    device     = args.device     or cfg["detection"]["device"]
    sequences  = args.sequences  or cfg["dataset"].get("sequences")
    eval_seqs  = cfg["dataset"].get("eval_sequences")
    classes    = cfg["dataset"]["class_names"]
    fps        = cfg["tracking"]["frame_rate"]
    visualize  = not (args.no_video or args.eval_only)

    console.rule("[bold cyan]KITTI Vehicle Tracker")
    console.print(f"  KITTI root : [green]{kitti_root}[/]")
    console.print(f"  Model      : [green]{model_path}[/]")
    console.print(f"  Device     : [green]{device}[/]")
    console.print(f"  Output     : [green]{output_dir}[/]")
    console.print(f"  Visualize  : [green]{visualize}[/]")
    console.rule()

    # ── Build components ──────────────────────────────────────────────────────
    console.print("[bold]Loading detector …[/]")
    detector = KITTIDetector(
        model_path=model_path,
        conf_threshold=cfg["detection"]["conf_threshold"],
        iou_threshold=cfg["detection"]["iou_threshold"],
        device=device,
        half_precision=cfg["detection"]["half_precision"],
        img_size=cfg["dataset"]["img_size"],
    )

    tracker = KITTITracker(
        track_thresh=cfg["tracking"]["track_thresh"],
        match_thresh=cfg["tracking"]["match_thresh"],
        track_buffer=cfg["tracking"]["track_buffer"],
        frame_rate=cfg["tracking"]["frame_rate"],
    )

    console.print("[bold]Loading dataset …[/]")
    dataset = KITTITrackingDataset(
        kitti_root=kitti_root,
        split="training",
        sequences=sequences,
        allowed_classes=classes,
        min_height=cfg["evaluation"]["min_height"],
        max_occlusion=cfg["evaluation"]["max_occlusion"],
        max_truncation=cfg["evaluation"]["max_truncation"],
    )

    console.print(
        f"  Found [cyan]{len(dataset)}[/] sequences: "
        f"{dataset.sequence_ids}"
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    console.print("\n[bold]Running pipeline …[/]")
    results_df = run_all_sequences(
        dataset=dataset,
        detector=detector,
        tracker=tracker,
        output_dir=Path(output_dir),
        eval_sequences=eval_seqs,
        fps=float(fps),
        iou_threshold=cfg["evaluation"]["iou_threshold"],
        classes=classes,
        visualize=visualize,
        demo_seq_id=args.demo_seq,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    console.print("\n")
    _print_results(results_df)

    csv_path = Path(output_dir) / "mota_results.csv"
    console.print(f"\n[green]Results saved to {csv_path}[/]")

    if visualize:
        demo_path = Path(output_dir) / "videos" / "tracking_demo.mp4"
        if demo_path.exists():
            console.print(f"[green]Demo video: {demo_path}[/]")


if __name__ == "__main__":
    main()
