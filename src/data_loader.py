"""
src/data_loader.py
──────────────────
KITTI Tracking dataset loader.

KITTI directory layout expected:
  data/kitti/
    training/
      image_02/<seq_id>/000000.png ...
      label_02/<seq_id>.txt
      calib/<seq_id>.txt

Label format (per line):
  frame  track_id  type  truncated  occluded  alpha
  bbox_left  bbox_top  bbox_right  bbox_bottom
  height  width  length  x  y  z  rotation_y  [score]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class KITTIBox:
    """Single ground-truth bounding box from a KITTI label file."""
    frame: int
    track_id: int
    cls: str               # e.g. "Car", "Pedestrian", "Cyclist"
    truncated: float       # 0.0–1.0
    occluded: int          # 0=visible, 1=partly, 2=largely, 3=unknown
    alpha: float
    x1: float
    y1: float
    x2: float
    y2: float
    # 3-D attributes (optional for 2-D tracking)
    height_3d: float = 0.0
    width_3d: float  = 0.0
    length_3d: float = 0.0
    x_3d: float      = 0.0
    y_3d: float      = 0.0
    z_3d: float      = 0.0
    rotation_y: float = 0.0
    score: float      = -1.0   # confidence, -1 if GT

    @property
    def bbox_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def bbox_xywh(self) -> np.ndarray:
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return np.array([self.x1, self.y1, w, h], dtype=np.float32)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def height_px(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class SequenceInfo:
    seq_id: int
    seq_name: str
    image_dir: Path
    label_path: Path
    n_frames: int
    image_shape: Tuple[int, int, int]   # (H, W, C)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_ignore_regions(
    label_path: Path,
    allowed_classes: Optional[List[str]] = None,
) -> Dict[int, np.ndarray]:
    """
    Parse all 'ignore' regions from a KITTI label file — i.e. every GT box
    whose class is NOT in `allowed_classes`.

    Why this matters
    ─────────────────
    KITTI labels several classes we don't explicitly evaluate: 'DontCare'
    (deliberately unlabeled background), but also 'Van', 'Truck', 'Tram',
    'Misc', 'Person_sitting'. If we only keep {Car, Pedestrian, Cyclist} in
    ground truth, a real Van/Truck GT box is not "ignored" — it's silently
    DELETED. A detector that correctly spots that van (mapped to our 'Car'
    class) now has no matching GT box at all, so it gets scored as a false
    positive despite being a genuinely correct detection.

    This function returns the union of literal 'DontCare' boxes AND any
    other-class box (Van, Truck, etc.) so the evaluator can exclude
    predictions overlapping ANY of them — matching how most KITTI-based
    work handles cross-class confusion between visually similar vehicle
    categories.

    Returns
    -------
    Dict[int, np.ndarray]
        frame_id -> (N, 4) array of [x1, y1, x2, y2] ignore-region boxes.
    """
    if allowed_classes is None:
        allowed_classes = ["Car", "Pedestrian", "Cyclist"]

    frames: Dict[int, List[List[float]]] = {}

    if not label_path.exists():
        return {}

    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue

            cls = parts[2]
            if cls in allowed_classes:
                continue   # this is a normal evaluated box, not an ignore region

            frame = int(parts[0])
            x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            frames.setdefault(frame, []).append([x1, y1, x2, y2])

    return {
        frame_id: np.array(boxes, dtype=np.float32)
        for frame_id, boxes in frames.items()
    }


def parse_dontcare_regions(label_path: Path) -> Dict[int, np.ndarray]:
    """
    Parse ONLY literal 'DontCare' regions from a KITTI label file.

    Kept for backward compatibility / explicit DontCare-only use cases.
    For full correctness, prefer `parse_ignore_regions()`, which also
    excludes Van/Truck/Tram/etc. — see its docstring for why this matters.

    Returns
    -------
    Dict[int, np.ndarray]
        frame_id -> (N, 4) array of [x1, y1, x2, y2] DontCare boxes.
    """
    frames: Dict[int, List[List[float]]] = {}

    if not label_path.exists():
        return {}

    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10 or parts[2] != "DontCare":
                continue

            frame = int(parts[0])
            x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            frames.setdefault(frame, []).append([x1, y1, x2, y2])

    return {
        frame_id: np.array(boxes, dtype=np.float32)
        for frame_id, boxes in frames.items()
    }


def parse_label_file(
    label_path: Path,
    allowed_classes: Optional[List[str]] = None,
    min_height: float = 0.0,
    max_occlusion: int = 3,
    max_truncation: float = 1.0,
) -> Dict[int, List[KITTIBox]]:
    """
    Parse a KITTI label file into a dict keyed by frame index.

    Returns
    -------
    Dict[int, List[KITTIBox]]
        frame_id -> list of KITTIBox objects passing the filters
    """
    if allowed_classes is None:
        allowed_classes = ["Car", "Pedestrian", "Cyclist"]

    frames: Dict[int, List[KITTIBox]] = {}

    if not label_path.exists():
        return frames

    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 17:
                continue

            frame     = int(parts[0])
            track_id  = int(parts[1])
            cls       = parts[2]
            truncated = float(parts[3])
            occluded  = int(parts[4])
            alpha     = float(parts[5])
            x1        = float(parts[6])
            y1        = float(parts[7])
            x2        = float(parts[8])
            y2        = float(parts[9])
            h3d       = float(parts[10])
            w3d       = float(parts[11])
            l3d       = float(parts[12])
            xc        = float(parts[13])
            yc        = float(parts[14])
            zc        = float(parts[15])
            ry        = float(parts[16])
            score     = float(parts[17]) if len(parts) > 17 else -1.0

            # ── Filters ──
            if cls not in allowed_classes:
                continue
            if occluded > max_occlusion:
                continue
            if truncated > max_truncation:
                continue
            height_px = max(0.0, y2 - y1)
            if height_px < min_height:
                continue

            box = KITTIBox(
                frame=frame, track_id=track_id, cls=cls,
                truncated=truncated, occluded=occluded, alpha=alpha,
                x1=x1, y1=y1, x2=x2, y2=y2,
                height_3d=h3d, width_3d=w3d, length_3d=l3d,
                x_3d=xc, y_3d=yc, z_3d=zc, rotation_y=ry, score=score,
            )

            frames.setdefault(frame, []).append(box)

    return frames


# ── Sequence loader ───────────────────────────────────────────────────────────

class KITTISequence:
    """
    Iterable over frames of a single KITTI tracking sequence.

    Usage
    -----
    seq = KITTISequence(seq_id=0, kitti_root=Path("data/kitti"), split="training")
    for frame_id, image, gt_boxes in seq:
        ...
    """

    def __init__(
        self,
        seq_id: int,
        kitti_root: Path,
        split: str = "training",
        allowed_classes: Optional[List[str]] = None,
        min_height: float = 0.0,
        max_occlusion: int = 3,
        max_truncation: float = 1.0,
    ):
        self.seq_id    = seq_id
        self.seq_name  = f"{seq_id:04d}"
        self.split     = split
        self.root      = Path(kitti_root)

        self.image_dir  = self.root / split / "image_02" / self.seq_name
        self.label_path = self.root / split / "label_02" / f"{self.seq_name}.txt"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")

        # Sorted list of image paths
        self._frames = sorted(self.image_dir.glob("*.png")) + \
                       sorted(self.image_dir.glob("*.jpg"))
        if not self._frames:
            raise ValueError(f"No images found in {self.image_dir}")

        # Parse GT labels (may be empty for test split)
        self.gt: Dict[int, List[KITTIBox]] = parse_label_file(
            self.label_path,
            allowed_classes=allowed_classes,
            min_height=min_height,
            max_occlusion=max_occlusion,
            max_truncation=max_truncation,
        )

        # Parse ignore regions — DontCare PLUS any GT box whose class isn't
        # in allowed_classes (e.g. Van/Truck when we only evaluate Car).
        # Used by the evaluator to exclude predictions on these regions from
        # being counted as false positives. See parse_ignore_regions() for
        # the full rationale.
        self.dontcare: Dict[int, np.ndarray] = parse_ignore_regions(
            self.label_path, allowed_classes=allowed_classes
        )

        # Cache image shape from first frame
        _img0 = cv2.imread(str(self._frames[0]))
        self.image_shape: Tuple[int, int, int] = _img0.shape  # (H, W, C)

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def info(self) -> SequenceInfo:
        return SequenceInfo(
            seq_id=self.seq_id,
            seq_name=self.seq_name,
            image_dir=self.image_dir,
            label_path=self.label_path,
            n_frames=self.n_frames,
            image_shape=self.image_shape,
        )

    def get_dontcare(self, frame_id: int) -> np.ndarray:
        """Return (N, 4) xyxy DontCare boxes for a frame, or empty (0,4) array."""
        return self.dontcare.get(frame_id, np.zeros((0, 4), dtype=np.float32))

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Generator[Tuple[int, np.ndarray, List[KITTIBox]], None, None]:
        for frame_id, img_path in enumerate(self._frames):
            image = cv2.imread(str(img_path))
            if image is None:
                raise IOError(f"Failed to read image: {img_path}")
            gt_boxes = self.gt.get(frame_id, [])
            yield frame_id, image, gt_boxes

    def get_frame(self, frame_id: int) -> Tuple[np.ndarray, List[KITTIBox]]:
        """Load a single frame by index."""
        img = cv2.imread(str(self._frames[frame_id]))
        if img is None:
            raise IOError(f"Failed to read frame {frame_id}")
        return img, self.gt.get(frame_id, [])


# ── Dataset (multi-sequence) ──────────────────────────────────────────────────

class KITTITrackingDataset:
    """
    Thin wrapper that discovers and loads all (or selected) sequences.

    Parameters
    ----------
    kitti_root   : Path to KITTI root directory.
    split        : "training" or "testing".
    sequences    : None = all, or list of int seq IDs.
    """

    def __init__(
        self,
        kitti_root: str | Path,
        split: str = "training",
        sequences: Optional[List[int]] = None,
        allowed_classes: Optional[List[str]] = None,
        min_height: float = 0.0,
        max_occlusion: int = 3,
        max_truncation: float = 1.0,
    ):
        self.root  = Path(kitti_root)
        self.split = split
        self._seq_kwargs = dict(
            kitti_root=self.root,
            split=split,
            allowed_classes=allowed_classes,
            min_height=min_height,
            max_occlusion=max_occlusion,
            max_truncation=max_truncation,
        )

        img_root = self.root / split / "image_02"
        if not img_root.exists():
            raise FileNotFoundError(f"KITTI image root not found: {img_root}")

        available = sorted(
            int(p.name) for p in img_root.iterdir()
            if p.is_dir() and re.fullmatch(r"\d{4}", p.name)
        )

        if sequences is not None:
            missing = [s for s in sequences if s not in available]
            if missing:
                raise ValueError(f"Sequences not found: {missing}")
            self._seq_ids = [s for s in sequences if s in available]
        else:
            self._seq_ids = available

    @property
    def sequence_ids(self) -> List[int]:
        return list(self._seq_ids)

    def __len__(self) -> int:
        return len(self._seq_ids)

    def __getitem__(self, idx: int) -> KITTISequence:
        return KITTISequence(seq_id=self._seq_ids[idx], **self._seq_kwargs)

    def iter_sequences(self) -> Generator[KITTISequence, None, None]:
        for seq_id in self._seq_ids:
            yield KITTISequence(seq_id=seq_id, **self._seq_kwargs)
