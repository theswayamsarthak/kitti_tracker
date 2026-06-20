"""
src/detector.py
───────────────
YOLO-based frame detector for KITTI classes.

Supports two modes:
  1. COCO-pretrained (default): maps COCO class indices -> KITTI class
     names (Car/Pedestrian/Cyclist) since the model never saw KITTI's
     actual labels.
  2. Fine-tuned (finetuned=True): the model was trained directly on KITTI
     data via scripts/convert_kitti_to_yolo.py + scripts/finetune_yolo.py,
     so it already outputs class ids 0=Car, 1=Pedestrian, 2=Cyclist
     natively — no remapping needed, used as-is.

Detection output is returned as supervision.Detections so it plugs
directly into the ByteTrack tracker without extra conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO


# ── COCO → KITTI mapping ──────────────────────────────────────────────────────
# Only used when finetuned=False (default). YOLOv8/11/26 pretrained on COCO;
# we remap the relevant classes onto KITTI's 3 evaluated classes.
# COCO ids: person=0, bicycle=1, car=2, motorbike=3, bus=5, truck=7
_COCO_TO_KITTI: Dict[int, str] = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Car",
    3: "Cyclist",   # motorbike → Cyclist (closest KITTI match)
    5: "Car",       # bus → Car
    7: "Car",       # truck → Car
}

_KITTI_CLASSES = ["Car", "Pedestrian", "Cyclist"]


class KITTIDetector:
    """
    Thin wrapper around Ultralytics YOLO that:
      - filters to vehicle-relevant classes
      - maps them to KITTI class names (COCO-pretrained mode only)
      - returns supervision.Detections

    Parameters
    ----------
    model_path      : Path to .pt weights. For COCO-pretrained mode, e.g.
                      "yolo26m.pt" (auto-downloaded). For fine-tuned mode,
                      the path to your checkpoint from
                      scripts/finetune_yolo.py (e.g.
                      "checkpoints/kitti_finetuned.pt").
    conf_threshold  : Minimum detection confidence (0–1).
    iou_threshold   : NMS IoU threshold (0–1).
    device          : "cpu", "cuda", "cuda:0", or "auto".
    half_precision  : Use FP16 on GPU if True.
    img_size        : Inference resolution (keeps aspect ratio).
    agnostic_nms    : If True, NMS suppresses overlapping boxes regardless
                      of predicted class. REQUIRED in COCO-pretrained mode
                      because we remap multiple COCO classes (car/bus/truck)
                      onto a single KITTI class (Car) — without this,
                      YOLO's default per-class NMS can let two overlapping
                      boxes survive (e.g. one labeled 'car', one labeled
                      'truck', for the SAME physical vehicle) since they
                      were different classes at NMS time. Also left on by
                      default for fine-tuned mode — harmless there since
                      the model's classes are already mutually exclusive,
                      but doesn't hurt.
    finetuned       : If True, skip COCO→KITTI remapping entirely. Assumes
                      the model was trained via convert_kitti_to_yolo.py,
                      which fixes class ids as 0=Car, 1=Pedestrian,
                      2=Cyclist — output is used directly.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26m.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float  = 0.45,
        device: str = "auto",
        half_precision: bool = True,
        img_size: int = 1280,
        agnostic_nms: bool = True,
        finetuned: bool = False,
    ):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device       = device
        self.conf         = conf_threshold
        self.iou          = iou_threshold
        self.img_size     = img_size
        self.half         = half_precision and (device != "cpu")
        self.agnostic_nms = agnostic_nms
        self.finetuned    = finetuned

        self.model = YOLO(str(model_path))

        # Class label array used by supervision (index → name)
        self._class_names = np.array(_KITTI_CLASSES)

        # COCO indices we care about — only relevant in COCO-pretrained mode.
        # In fine-tuned mode, `classes` filter is omitted entirely (model
        # only has 3 classes anyway, all of which we want).
        self._coco_classes = list(_COCO_TO_KITTI.keys())

        # class name → stable integer id for supervision
        self._kitti_class_id: Dict[str, int] = {
            name: i for i, name in enumerate(_KITTI_CLASSES)
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a single BGR frame (OpenCV format).

        Returns
        -------
        sv.Detections
            xyxy      : (N, 4) float32
            confidence: (N,)   float32
            class_id  : (N,)   int        KITTI class index (0=Car,1=Ped,2=Cyc)
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=None if self.finetuned else self._coco_classes,
            imgsz=self.img_size,
            device=self.device,
            half=self.half,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()

        boxes_xyxy  = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        raw_ids     = result.boxes.cls.cpu().numpy().astype(int)

        if self.finetuned:
            # Model already outputs KITTI class ids directly — no remap.
            return sv.Detections(
                xyxy=boxes_xyxy,
                confidence=confidences,
                class_id=raw_ids,
            )

        # ── COCO-pretrained mode: map COCO → KITTI, drop unknowns ──────────
        kitti_ids   = np.array(
            [self._kitti_class_id[_COCO_TO_KITTI[c]] for c in raw_ids
             if c in _COCO_TO_KITTI],
            dtype=int,
        )
        keep = np.array(
            [i for i, c in enumerate(raw_ids) if c in _COCO_TO_KITTI],
            dtype=int,
        )

        if len(keep) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=boxes_xyxy[keep],
            confidence=confidences[keep],
            class_id=kitti_ids,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """Run detection on a list of frames (batch inference)."""
        return [self.detect(f) for f in frames]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def class_names(self) -> List[str]:
        return _KITTI_CLASSES

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(_KITTI_CLASSES):
            return _KITTI_CLASSES[class_id]
        return "Unknown"
