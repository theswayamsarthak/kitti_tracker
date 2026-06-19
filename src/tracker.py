"""
src/tracker.py
──────────────
ByteTrack multi-object tracker via the supervision library.

supervision >= 0.21 ships ByteTrack natively; no separate install needed.
The tracker maintains persistent integer track IDs across frames and
stores a trajectory history for each active track.

Usage
-----
tracker = KITTITracker(track_thresh=0.25, match_thresh=0.8,
                        track_buffer=30, frame_rate=10)

for frame_id, image, _ in sequence:
    detections = detector.detect(image)
    tracks     = tracker.update(detections)
    # tracks is a TrackerOutput with .detections (sv.Detections) + .histories
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv


@dataclass
class TrackerOutput:
    """
    Output from a single tracker update call.

    Attributes
    ----------
    detections  : sv.Detections with tracker_id populated
    histories   : dict mapping track_id → deque of (cx, cy) centroids
    frame_id    : frame index this update was called on
    """
    detections: sv.Detections
    histories: Dict[int, deque]
    frame_id: int

    @property
    def track_ids(self) -> np.ndarray:
        if self.detections.tracker_id is None:
            return np.array([], dtype=int)
        return self.detections.tracker_id

    @property
    def n_active(self) -> int:
        return len(self.detections)


class KITTITracker:
    """
    Stateful ByteTrack wrapper.

    Wraps supervision.ByteTrack and adds:
      - Per-track class identity (majority vote across frames)
      - Trajectory history (centroid trail)
      - Clean reset between sequences

    Parameters
    ----------
    track_thresh    : Detection confidence threshold for high-score tracks.
    match_thresh    : IoU threshold for track-detection association.
    track_buffer    : Max frames a track survives without a match.
    frame_rate      : Source video FPS (used to scale track_buffer).
    history_len     : Max centroid trail length per track.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        track_buffer: int   = 30,
        frame_rate: int     = 10,
        history_len: int    = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate   = frame_rate
        self.history_len  = history_len

        self._init_tracker()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_tracker(self) -> None:
        """(Re-)initialise a fresh ByteTrack instance."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
        )
        # track_id → deque of (cx, cy)
        self._histories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_len)
        )
        # track_id → list of class_id votes
        self._class_votes: Dict[int, List[int]] = defaultdict(list)

    def reset(self) -> None:
        """Reset tracker state — call between sequences."""
        self._init_tracker()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: sv.Detections,
        frame_id: int = 0,
    ) -> TrackerOutput:
        """
        Feed a frame's detections to ByteTrack and return tracked results.

        Parameters
        ----------
        detections  : Output of KITTIDetector.detect() — sv.Detections.
        frame_id    : Current frame index (used only for bookkeeping).

        Returns
        -------
        TrackerOutput
        """
        if len(detections) == 0:
            return TrackerOutput(
                detections=sv.Detections.empty(),
                histories=dict(self._histories),
                frame_id=frame_id,
            )

        # ByteTrack.update returns sv.Detections with tracker_id filled in
        tracked: sv.Detections = self._tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked) == 0:
            return TrackerOutput(
                detections=sv.Detections.empty(),
                histories=dict(self._histories),
                frame_id=frame_id,
            )

        # ── Update histories & class votes ───────────────────────────────────
        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self._histories[int(tid)].append((cx, cy))

            if tracked.class_id is not None:
                self._class_votes[int(tid)].append(int(tracked.class_id[i]))

        return TrackerOutput(
            detections=tracked,
            histories={tid: deque(hist) for tid, hist in self._histories.items()},
            frame_id=frame_id,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def stable_class(self, track_id: int) -> Optional[int]:
        """
        Return majority-vote class for a track (more stable than per-frame).
        Returns None if the track has no class votes.
        """
        votes = self._class_votes.get(track_id)
        if not votes:
            return None
        return int(np.bincount(votes).argmax())

    def get_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Return centroid trail for a track as a list of (cx, cy)."""
        return list(self._histories.get(track_id, []))
