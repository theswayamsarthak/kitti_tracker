"""
src/visualizer.py
─────────────────
Renders annotated tracking frames and encodes them to MP4.

Features
────────
- Color-coded bounding boxes per class
- Track ID labels
- Confidence scores
- Trajectory trail (centroid history)
- Frame counter + sequence info overlay
- Writes MP4 via OpenCV VideoWriter (H.264 / XVID fallback)
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


# ── Color palette (BGR) ──────────────────────────────────────────────────────

_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Car":        (100, 200,   0),   # green
    "Pedestrian": (255, 150,  50),   # blue-ish
    "Cyclist":    ( 50, 150, 255),   # orange
}
_DEFAULT_COLOR = (180, 180, 180)

# 20 visually distinct track-ID colors for the trail
_TRACK_PALETTE = [
    (  0, 200, 100), (  0, 100, 255), (255, 180,  30), (200,  50, 200),
    (  0, 220, 220), (255,  60,  60), (150, 255, 100), ( 80,  80, 255),
    (255, 100, 180), ( 30, 200, 180), (200, 200,  50), (100,  50, 255),
    (255, 130,  80), ( 40, 200,  40), (200,  40,  40), ( 50, 180, 255),
    (255, 200, 100), (150,  50,  50), ( 80, 200, 200), (200, 150, 255),
]

def _track_color(track_id: int) -> Tuple[int, int, int]:
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


# ── VideoWriter factory ───────────────────────────────────────────────────────

def _make_writer(
    out_path: Path,
    fps: float,
    frame_size: Tuple[int, int],   # (W, H)
) -> cv2.VideoWriter:
    """Try H.264 first, fall back to XVID."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fourcc_str in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(
            str(out_path), fourcc, fps, frame_size
        )
        if writer.isOpened():
            logger.info(f"VideoWriter opened with codec '{fourcc_str}'")
            return writer

    raise RuntimeError(f"Could not open VideoWriter for {out_path}")


# ── Core drawing helpers ──────────────────────────────────────────────────────

def _draw_box_and_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> None:
    """Draw a filled-header bounding box with class/track label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    font      = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_y1  = max(y1 - th - baseline - 4, 0)
    label_y2  = y1
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, label_y2 - baseline),
        font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_trail(
    frame: np.ndarray,
    history: deque,
    color: Tuple[int, int, int],
    max_len: int = 30,
) -> None:
    """Draw fading centroid trail for one track."""
    pts = list(history)[-max_len:]
    for i in range(1, len(pts)):
        alpha  = i / len(pts)
        radius = max(1, int(2 * alpha))
        c = tuple(int(v * alpha) for v in color)
        cv2.line(
            frame,
            (int(pts[i - 1][0]), int(pts[i - 1][1])),
            (int(pts[i][0]),     int(pts[i][1])),
            c, radius, cv2.LINE_AA,
        )


def _draw_hud(
    frame: np.ndarray,
    frame_id: int,
    seq_name: str,
    n_tracks: int,
) -> None:
    """Draw semi-transparent HUD in top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Seq: {seq_name}  Frame: {frame_id:04d}",
                (8, 20), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active tracks: {n_tracks}",
                (8, 42), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


# ── FrameAnnotator ────────────────────────────────────────────────────────────

class FrameAnnotator:
    """
    Annotates a single frame with detections + tracking info.

    Parameters
    ----------
    class_names     : List of class name strings (index → name).
    box_thickness   : Bounding box line width.
    font_scale      : Label font scale.
    draw_trails     : Whether to draw centroid trail.
    trail_len       : Max frames of history to draw.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        box_thickness: int   = 2,
        font_scale: float    = 0.55,
        draw_trails: bool    = True,
        trail_len: int       = 30,
    ):
        self.class_names   = class_names or ["Car", "Pedestrian", "Cyclist"]
        self.box_thickness = box_thickness
        self.font_scale    = font_scale
        self.draw_trails   = draw_trails
        self.trail_len     = trail_len

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        histories: Dict[int, deque],
        frame_id: int = 0,
        seq_name: str = "",
    ) -> np.ndarray:
        """
        Return an annotated copy of frame.

        Parameters
        ----------
        frame      : BGR image (not modified in-place).
        detections : sv.Detections with tracker_id populated.
        histories  : dict of track_id → deque of centroids.
        frame_id   : Current frame index for HUD.
        seq_name   : Sequence name for HUD.
        """
        out = frame.copy()

        if detections.tracker_id is None or len(detections) == 0:
            _draw_hud(out, frame_id, seq_name, 0)
            return out

        # ── Draw trails first (behind boxes) ──────────────────────────────
        if self.draw_trails:
            for tid in detections.tracker_id:
                hist = histories.get(int(tid))
                if hist:
                    color = _track_color(int(tid))
                    _draw_trail(out, hist, color, self.trail_len)

        # ── Draw boxes + labels ───────────────────────────────────────────
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            tid   = int(detections.tracker_id[i])
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            conf   = float(detections.confidence[i]) if detections.confidence is not None else 0.0

            cls_name  = self.class_names[cls_id] if cls_id < len(self.class_names) else "?"
            color     = _CLASS_COLORS.get(cls_name, _DEFAULT_COLOR)
            label     = f"{cls_name[0]} #{tid} {conf:.2f}"

            _draw_box_and_label(
                out, x1, y1, x2, y2, color, label,
                thickness=self.box_thickness,
                font_scale=self.font_scale,
            )

        _draw_hud(out, frame_id, seq_name, len(detections))
        return out


# ── VideoWriter ───────────────────────────────────────────────────────────────

class TrackingVideoWriter:
    """
    Context-managed MP4 writer.

    Usage
    -----
    annotator = FrameAnnotator(...)

    with TrackingVideoWriter("outputs/demo.mp4", fps=10) as vw:
        for frame_id, image, _ in sequence:
            annotated = annotator.annotate(image, tracks.detections,
                                           tracks.histories, frame_id, "0000")
            vw.write(annotated)
    """

    def __init__(self, out_path: str | Path, fps: float = 10.0):
        self.out_path = Path(out_path)
        self.fps      = fps
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            self._writer = _make_writer(self.out_path, self.fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "TrackingVideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
