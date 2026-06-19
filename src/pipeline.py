"""
src/pipeline.py
───────────────
Orchestrates the full detect → track → evaluate → visualize pipeline
for one or more KITTI sequences.

Entry points
────────────
run_sequence()     — process a single KITTISequence, return metrics
run_all_sequences()— process a full KITTITrackingDataset
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import KITTISequence, KITTITrackingDataset
from src.detector import KITTIDetector
from src.evaluator import MOTAEvaluator
from src.tracker import KITTITracker
from src.visualizer import FrameAnnotator, TrackingVideoWriter

logger = logging.getLogger(__name__)


# ── Single-sequence runner ────────────────────────────────────────────────────

def run_sequence(
    sequence: KITTISequence,
    detector: KITTIDetector,
    tracker: KITTITracker,
    evaluator: Optional[MOTAEvaluator] = None,
    output_video_path: Optional[Path] = None,
    fps: float = 10.0,
    visualize: bool = True,
    iou_threshold: float = 0.5,
    classes: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Run the full pipeline on one KITTI sequence.

    Parameters
    ----------
    sequence          : A KITTISequence instance.
    detector          : Initialised KITTIDetector.
    tracker           : Initialised KITTITracker (will be reset).
    evaluator         : Optional MOTAEvaluator (will be reset).
    output_video_path : If provided, writes an annotated MP4 here.
    fps               : Output video FPS.
    visualize         : Whether to run visualizer (can disable for speed).
    iou_threshold     : IoU threshold for evaluation.
    classes           : Class names list (must match detector + evaluator).

    Returns
    -------
    (metrics_df, elapsed_seconds)
        metrics_df is None if no evaluator was provided.
    """
    classes = classes or ["Car", "Pedestrian", "Cyclist"]

    tracker.reset()
    if evaluator:
        evaluator.reset()

    annotator = FrameAnnotator(
        class_names=classes,
        box_thickness=2,
        font_scale=0.55,
        draw_trails=True,
        trail_len=30,
    )

    seq_name = sequence.seq_name
    logger.info(f"Processing sequence {seq_name} ({sequence.n_frames} frames)")

    t_start = time.perf_counter()

    video_ctx = (
        TrackingVideoWriter(output_video_path, fps=fps)
        if (visualize and output_video_path is not None)
        else _NullVideoWriter()
    )

    with video_ctx as vw:
        for frame_id, image, gt_boxes in tqdm(
            sequence,
            total=sequence.n_frames,
            desc=f"Seq {seq_name}",
            unit="frame",
            leave=False,
        ):
            # ── Detect ────────────────────────────────────────────────────
            detections = detector.detect(image)

            # ── Track ─────────────────────────────────────────────────────
            tracks = tracker.update(detections, frame_id=frame_id)

            # ── Evaluate ──────────────────────────────────────────────────
            if evaluator is not None:
                pred_xyxy = (
                    tracks.detections.xyxy
                    if len(tracks.detections) > 0
                    else np.zeros((0, 4), dtype=np.float32)
                )
                pred_ids = (
                    tracks.detections.tracker_id
                    if tracks.detections.tracker_id is not None
                    else np.array([], dtype=int)
                )
                pred_cls = (
                    tracks.detections.class_id
                    if tracks.detections.class_id is not None
                    else np.array([], dtype=int)
                )
                evaluator.update(
                    frame_id=frame_id,
                    gt_boxes=gt_boxes,
                    pred_xyxy=pred_xyxy,
                    pred_ids=pred_ids,
                    pred_classes=pred_cls,
                    dontcare_xyxy=sequence.get_dontcare(frame_id),
                )

            # ── Visualize ─────────────────────────────────────────────────
            if visualize:
                annotated = annotator.annotate(
                    image,
                    tracks.detections,
                    tracks.histories,
                    frame_id=frame_id,
                    seq_name=seq_name,
                )
                vw.write(annotated)

    elapsed = time.perf_counter() - t_start
    fps_achieved = sequence.n_frames / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"Seq {seq_name}: {sequence.n_frames} frames in {elapsed:.1f}s "
        f"({fps_achieved:.1f} fps)"
    )

    metrics_df = evaluator.compute() if evaluator is not None else None
    return metrics_df, elapsed


# ── Multi-sequence runner ─────────────────────────────────────────────────────

def run_all_sequences(
    dataset: KITTITrackingDataset,
    detector: KITTIDetector,
    tracker: KITTITracker,
    output_dir: Path,
    eval_sequences: Optional[List[int]] = None,
    fps: float = 10.0,
    iou_threshold: float = 0.5,
    classes: Optional[List[str]] = None,
    visualize: bool = True,
    demo_seq_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the pipeline over all sequences in a dataset.

    Parameters
    ----------
    dataset         : KITTITrackingDataset instance.
    detector        : Shared KITTIDetector (model loaded once).
    tracker         : KITTITracker (reset per sequence).
    output_dir      : Root for output videos and results.
    eval_sequences  : Subset of seq IDs to evaluate; None = all.
    fps             : Video FPS.
    iou_threshold   : IoU for evaluation.
    classes         : Class names.
    visualize       : Write annotated videos.
    demo_seq_id     : If set, this sequence's video is named tracking_demo.mp4.

    Returns
    -------
    pd.DataFrame with per-sequence MOTA results.
    """
    classes       = classes or ["Car", "Pedestrian", "Cyclist"]
    output_dir    = Path(output_dir)
    videos_dir    = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for seq in dataset.iter_sequences():
        seq_id = seq.seq_id

        should_eval  = (eval_sequences is None) or (seq_id in eval_sequences)
        evaluator    = MOTAEvaluator(
            iou_threshold=iou_threshold, classes=classes
        ) if should_eval else None

        is_demo      = (demo_seq_id is None) or (seq_id == demo_seq_id)
        video_name   = "tracking_demo.mp4" if is_demo else f"seq_{seq.seq_name}.mp4"
        video_path   = videos_dir / video_name if visualize else None

        metrics_df, elapsed = run_sequence(
            sequence=seq,
            detector=detector,
            tracker=tracker,
            evaluator=evaluator,
            output_video_path=video_path,
            fps=fps,
            visualize=visualize,
            iou_threshold=iou_threshold,
            classes=classes,
        )

        row: Dict = {
            "seq_id":  seq_id,
            "seq_name": seq.seq_name,
            "n_frames": seq.n_frames,
            "time_s":   round(elapsed, 2),
            "fps":      round(seq.n_frames / elapsed, 1) if elapsed > 0 else 0,
        }

        if metrics_df is not None and "All" in metrics_df.index:
            for col in ["mota", "motp", "idf1", "num_switches",
                        "num_false_positives", "num_misses", "recall", "precision"]:
                if col in metrics_df.columns:
                    row[col] = metrics_df.loc["All", col]

        all_results.append(row)

        if metrics_df is not None:
            logger.info(
                f"Seq {seq.seq_name} | "
                f"MOTA={row.get('mota', 'N/A'):.2f}%  "
                f"IDF1={row.get('idf1', 'N/A'):.2f}%  "
                f"IDS={row.get('num_switches', 'N/A')}"
            )

    results_df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = output_dir / "mota_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    return results_df


# ── Null context manager ──────────────────────────────────────────────────────

class _NullVideoWriter:
    """No-op writer used when visualize=False."""
    def write(self, _): pass
    def release(self): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass
