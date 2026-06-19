"""
src/evaluator.py
────────────────
MOTA / MOTP evaluation against KITTI ground truth.

Uses py-motmetrics (motmetrics package) — the industry standard
for multi-object tracking evaluation.

Metrics computed
────────────────
MOTA   - Multi-Object Tracking Accuracy  (higher is better)
MOTP   - Multi-Object Tracking Precision (lower dist is better)
IDF1   - ID F1 Score                     (higher is better)
FP     - False Positives
FN     - False Negatives
IDS    - Identity Switches
Recall - Detection recall
Prec   - Detection precision
MT     - Mostly Tracked trajectories (%)
ML     - Mostly Lost trajectories (%)

Usage
─────
evaluator = MOTAEvaluator(iou_threshold=0.5)

for frame_id, _, gt_boxes in sequence:
    evaluator.update(
        frame_id=frame_id,
        gt_boxes=gt_boxes,
        pred_xyxy=tracked.detections.xyxy,
        pred_ids=tracked.detections.tracker_id,
        pred_classes=tracked.detections.class_id,
    )

summary = evaluator.compute()
print(summary)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import motmetrics as mm
import numpy as np
import pandas as pd

from src.data_loader import KITTIBox


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou_matrix(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.

    Parameters
    ----------
    gt_xyxy   : (M, 4) array of ground-truth boxes
    pred_xyxy : (N, 4) array of prediction boxes

    Returns
    -------
    (M, N) float32 IoU matrix
    """
    if gt_xyxy.shape[0] == 0 or pred_xyxy.shape[0] == 0:
        return np.zeros((gt_xyxy.shape[0], pred_xyxy.shape[0]), dtype=np.float32)

    # Expand dims for broadcasting
    g = gt_xyxy[:, None, :]    # (M, 1, 4)
    p = pred_xyxy[None, :, :]  # (1, N, 4)

    inter_x1 = np.maximum(g[..., 0], p[..., 0])
    inter_y1 = np.maximum(g[..., 1], p[..., 1])
    inter_x2 = np.minimum(g[..., 2], p[..., 2])
    inter_y2 = np.minimum(g[..., 3], p[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])

    union = area_g + area_p - inter
    iou   = np.where(union > 0, inter / union, 0.0)

    return iou.astype(np.float32)


def _iou_to_distance(iou: np.ndarray) -> np.ndarray:
    """Convert IoU to distance matrix (motmetrics expects distances, not IoU)."""
    return 1.0 - iou


# ── Per-class accumulator ─────────────────────────────────────────────────────

class _ClassAccumulator:
    """
    Wraps a motmetrics accumulator for a single class.
    """

    def __init__(self, class_name: str, iou_threshold: float = 0.5):
        self.class_name    = class_name
        self.iou_threshold = iou_threshold
        self.acc           = mm.MOTAccumulator(auto_id=True)

    def update(
        self,
        gt_ids: List[int],
        gt_xyxy: np.ndarray,
        pred_ids: List[int],
        pred_xyxy: np.ndarray,
    ) -> None:
        """
        Add one frame to the accumulator.

        gt_ids / pred_ids : list of integer object IDs
        gt_xyxy / pred_xyxy: (N, 4) float arrays
        """
        if len(gt_xyxy) == 0 and len(pred_xyxy) == 0:
            self.acc.update([], [], [])
            return

        if len(gt_xyxy) == 0:
            self.acc.update([], pred_ids, [])
            return

        if len(pred_xyxy) == 0:
            self.acc.update(gt_ids, [], [])
            return

        iou  = _iou_matrix(
            np.asarray(gt_xyxy, dtype=np.float32),
            np.asarray(pred_xyxy, dtype=np.float32),
        )
        dist = _iou_to_distance(iou)
        # Threshold: distances > (1 - iou_threshold) are invalid matches
        dist[dist > (1.0 - self.iou_threshold)] = np.nan

        self.acc.update(gt_ids, pred_ids, dist)


# ── Main evaluator ────────────────────────────────────────────────────────────

class MOTAEvaluator:
    """
    Frame-level streaming MOTA evaluator.

    Call .update() for each frame, then .compute() at the end.

    Parameters
    ----------
    iou_threshold  : IoU threshold for TP / FP classification.
    classes        : Classes to evaluate (each gets its own accumulator + combined).
    """

    METRICS = [
        "num_frames", "mota", "motp", "idf1",
        "num_objects",
        "num_predictions",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "recall",
        "precision",
        "mostly_tracked",
        "mostly_lost",
        "num_fragmentations",
    ]

    def __init__(
        self,
        iou_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
    ):
        self.iou_threshold = iou_threshold
        self.classes       = classes or ["Car", "Pedestrian", "Cyclist"]

        self._class_to_id = {c: i for i, c in enumerate(self.classes)}

        # One accumulator per class + one combined
        self._accs: Dict[str, _ClassAccumulator] = {
            cls: _ClassAccumulator(cls, iou_threshold)
            for cls in self.classes
        }
        self._accs["All"] = _ClassAccumulator("All", iou_threshold)

        self._n_frames = 0

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(
        self,
        frame_id: int,
        gt_boxes: List[KITTIBox],
        pred_xyxy: np.ndarray,
        pred_ids: np.ndarray,
        pred_classes: Optional[np.ndarray] = None,
        dontcare_xyxy: Optional[np.ndarray] = None,
        dontcare_iou_threshold: float = 0.5,
    ) -> None:
        """
        Record one frame's ground truth vs predictions.

        Parameters
        ----------
        frame_id    : Frame index (for bookkeeping).
        gt_boxes    : List of KITTIBox GT annotations.
        pred_xyxy   : (N, 4) predicted boxes.
        pred_ids    : (N,) integer track IDs.
        pred_classes: (N,) integer class IDs (0=Car,1=Ped,2=Cyc).
                      If None, all predictions treated as class-agnostic.
        dontcare_xyxy : (M, 4) KITTI 'DontCare' region boxes for this frame.
                      Per KITTI's official evaluation protocol, any
                      prediction overlapping a DontCare region above
                      `dontcare_iou_threshold` is excluded entirely — not
                      counted as a false positive. These regions mark
                      objects too distant/small/ambiguous to label, not
                      "this isn't an object" — without this filter,
                      correctly-detected background vehicles get unfairly
                      penalized as FPs.
        dontcare_iou_threshold : IoU threshold for the DontCare exclusion.
        """
        self._n_frames += 1

        pred_xyxy = np.asarray(pred_xyxy, dtype=np.float32) \
                    if pred_xyxy is not None and len(pred_xyxy) > 0 \
                    else np.zeros((0, 4), dtype=np.float32)
        pred_ids = np.asarray(pred_ids) if pred_ids is not None else np.array([], dtype=int)
        pred_classes = np.asarray(pred_classes) if pred_classes is not None else None

        # ── Filter out predictions overlapping DontCare regions ───────────────
        if dontcare_xyxy is not None and len(dontcare_xyxy) > 0 and len(pred_xyxy) > 0:
            iou = _iou_matrix(np.asarray(dontcare_xyxy, dtype=np.float32), pred_xyxy)
            max_iou_per_pred = iou.max(axis=0) if iou.shape[0] > 0 else np.zeros(len(pred_xyxy))
            keep = max_iou_per_pred < dontcare_iou_threshold

            pred_xyxy = pred_xyxy[keep]
            pred_ids  = pred_ids[keep] if len(pred_ids) > 0 else pred_ids
            if pred_classes is not None and len(pred_classes) > 0:
                pred_classes = pred_classes[keep]

        # ── All-class combined ────────────────────────────────────────────────
        all_gt_ids   = [b.track_id for b in gt_boxes]
        all_gt_xyxy  = np.array([b.bbox_xyxy for b in gt_boxes], dtype=np.float32) \
                       if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        all_pred_ids = list(pred_ids) if len(pred_ids) > 0 else []
        all_pred_xyxy= pred_xyxy

        self._accs["All"].update(
            all_gt_ids, all_gt_xyxy, all_pred_ids, all_pred_xyxy
        )

        # ── Per-class ────────────────────────────────────────────────────────
        for cls in self.classes:
            gt_cls = [b for b in gt_boxes if b.cls == cls]
            gt_ids_c   = [b.track_id for b in gt_cls]
            gt_xyxy_c  = np.array([b.bbox_xyxy for b in gt_cls], dtype=np.float32) \
                         if gt_cls else np.zeros((0, 4), dtype=np.float32)

            if pred_classes is not None and len(pred_classes) > 0:
                cls_id   = self._class_to_id.get(cls, -1)
                mask     = pred_classes == cls_id
                pred_ids_c  = list(pred_ids[mask]) if len(pred_ids) > 0 else []
                pred_xyxy_c = pred_xyxy[mask] if len(pred_xyxy) > 0 \
                              else np.zeros((0, 4), dtype=np.float32)
            else:
                # class-agnostic mode — same preds for every class
                pred_ids_c  = all_pred_ids
                pred_xyxy_c = all_pred_xyxy

            self._accs[cls].update(
                gt_ids_c, gt_xyxy_c, pred_ids_c, pred_xyxy_c
            )

    # ── Compute summary ───────────────────────────────────────────────────────

    def compute(self) -> pd.DataFrame:
        """
        Compute and return a DataFrame with one row per class + "All".

        Columns match self.METRICS.
        """
        mh      = mm.metrics.create()
        names   = list(self._accs.keys())
        accs    = [self._accs[n].acc for n in names]

        summary = mh.compute_many(
            accs,
            metrics=self.METRICS,
            names=names,
            generate_overall=False,
        )

        # Convert MOTP from distance to IoU-like (1 - dist) for readability
        if "motp" in summary.columns:
            summary["motp"] = (1.0 - summary["motp"]).clip(0.0, 1.0)

        # Pretty-format percentage columns
        for col in ["mota", "motp", "idf1", "recall", "precision"]:
            if col in summary.columns:
                summary[col] = (summary[col] * 100).round(2)

        return summary

    def compute_mota(self) -> float:
        """Quick scalar: return overall MOTA (0–100 scale)."""
        df = self.compute()
        if "All" in df.index and "mota" in df.columns:
            return float(df.loc["All", "mota"])
        return 0.0

    def reset(self) -> None:
        """Reset all accumulators (call between sequences)."""
        for acc in self._accs.values():
            acc.acc = mm.MOTAccumulator(auto_id=True)
        self._n_frames = 0
