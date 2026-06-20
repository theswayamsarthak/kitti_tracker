"""
tests/test_pipeline.py
──────────────────────
Unit tests that run without KITTI data or GPU.
Uses synthetic data to test all core logic paths.
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.data_loader import KITTIBox, parse_dontcare_regions, parse_ignore_regions, parse_label_file
from src.evaluator import MOTAEvaluator, _iou_matrix


# ── IoU helpers ───────────────────────────────────────────────────────────────

class TestIoUMatrix:
    def test_perfect_overlap(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou   = _iou_matrix(boxes, boxes)
        assert iou.shape == (1, 1)
        assert abs(iou[0, 0] - 1.0) < 1e-5

    def test_no_overlap(self):
        a = np.array([[0,  0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 0, 30, 10]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 0, 15, 10]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        # intersection = 5*10=50, union = 150
        expected = 50.0 / 150.0
        assert abs(iou[0, 0] - expected) < 1e-4

    def test_empty_gt(self):
        gt   = np.zeros((0, 4), dtype=np.float32)
        pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou  = _iou_matrix(gt, pred)
        assert iou.shape == (0, 1)

    def test_empty_pred(self):
        gt   = np.array([[0, 0, 10, 10]], dtype=np.float32)
        pred = np.zeros((0, 4), dtype=np.float32)
        iou  = _iou_matrix(gt, pred)
        assert iou.shape == (1, 0)

    def test_batch(self):
        gt   = np.array([[0,0,10,10],[20,20,30,30]], dtype=np.float32)
        pred = np.array([[0,0,10,10],[5,5,15,15],[20,20,30,30]], dtype=np.float32)
        iou  = _iou_matrix(gt, pred)
        assert iou.shape == (2, 3)
        assert abs(iou[0, 0] - 1.0) < 1e-5
        assert abs(iou[1, 2] - 1.0) < 1e-5
        assert iou[0, 2] == 0.0
        assert iou[1, 0] == 0.0


# ── KITTIBox ──────────────────────────────────────────────────────────────────

class TestKITTIBox:
    def _box(self, x1=10, y1=20, x2=50, y2=80):
        return KITTIBox(
            frame=0, track_id=1, cls="Car",
            truncated=0.0, occluded=0, alpha=0.0,
            x1=x1, y1=y1, x2=x2, y2=y2,
        )

    def test_bbox_xyxy(self):
        b = self._box()
        np.testing.assert_array_equal(b.bbox_xyxy, [10, 20, 50, 80])

    def test_bbox_xywh(self):
        b = self._box()
        np.testing.assert_array_equal(b.bbox_xywh, [10, 20, 40, 60])

    def test_area(self):
        b = self._box()
        assert b.area == 40 * 60

    def test_height_px(self):
        b = self._box()
        assert b.height_px == 60.0


# ── Label file parser ─────────────────────────────────────────────────────────

class TestParseLabelFile:
    def _write_label(self, tmp_path: Path, lines: list) -> Path:
        p = tmp_path / "0000.txt"
        p.write_text("\n".join(lines))
        return p

    def test_basic_parse(self, tmp_path):
        label = (
            "0 1 Car 0.0 0 0.0 100.0 200.0 300.0 400.0 "
            "1.5 1.8 4.0 0.5 1.7 20.0 0.1"
        )
        p = self._write_label(tmp_path, [label])
        frames = parse_label_file(p)
        assert 0 in frames
        assert len(frames[0]) == 1
        b = frames[0][0]
        assert b.cls == "Car"
        assert b.track_id == 1
        assert b.x1 == 100.0
        assert b.y2 == 400.0

    def test_class_filter(self, tmp_path):
        lines = [
            "0 1 Car 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0",
            "0 2 Pedestrian 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0",
            "0 3 Van 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0",
        ]
        p = self._write_label(tmp_path, lines)
        frames = parse_label_file(p, allowed_classes=["Car"])
        assert len(frames[0]) == 1
        assert frames[0][0].cls == "Car"

    def test_occlusion_filter(self, tmp_path):
        # occluded=3 should be excluded when max_occlusion=2
        lines = [
            "0 1 Car 0.0 3 0.0 0 0 100 100 1 1 1 0 0 0 0",
            "0 2 Car 0.0 1 0.0 0 0 100 100 1 1 1 0 0 0 0",
        ]
        p = self._write_label(tmp_path, lines)
        frames = parse_label_file(p, max_occlusion=2)
        assert len(frames[0]) == 1
        assert frames[0][0].track_id == 2

    def test_min_height_filter(self, tmp_path):
        # box height = y2 - y1 = 100 - 90 = 10 px (below min_height=25)
        lines = [
            "0 1 Car 0.0 0 0.0 0 90 100 100 1 1 1 0 0 0 0",  # h=10
            "0 2 Car 0.0 0 0.0 0 50 100 100 1 1 1 0 0 0 0",  # h=50
        ]
        p = self._write_label(tmp_path, lines)
        frames = parse_label_file(p, min_height=25)
        assert len(frames[0]) == 1
        assert frames[0][0].track_id == 2

    def test_empty_file(self, tmp_path):
        p = self._write_label(tmp_path, [])
        frames = parse_label_file(p)
        assert frames == {}

    def test_missing_file(self, tmp_path):
        frames = parse_label_file(tmp_path / "nonexistent.txt")
        assert frames == {}

    def test_multi_frame(self, tmp_path):
        lines = [
            f"{i} 1 Car 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0"
            for i in range(5)
        ]
        p = self._write_label(tmp_path, lines)
        frames = parse_label_file(p)
        assert set(frames.keys()) == {0, 1, 2, 3, 4}


class TestParseDontCareRegions:
    def _write_label(self, tmp_path: Path, lines: list) -> Path:
        p = tmp_path / "0000.txt"
        p.write_text("\n".join(lines))
        return p

    def test_extracts_dontcare(self, tmp_path):
        lines = [
            "0 1 Car 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0",
            "0 -1 DontCare -1 -1 -10 490 490 550 540 -1 -1 -1 -1 -1 -1 -1",
        ]
        p = self._write_label(tmp_path, lines)
        dontcare = parse_dontcare_regions(p)
        assert 0 in dontcare
        assert dontcare[0].shape == (1, 4)
        np.testing.assert_array_almost_equal(dontcare[0][0], [490, 490, 550, 540])

    def test_no_dontcare_in_file(self, tmp_path):
        lines = ["0 1 Car 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0"]
        p = self._write_label(tmp_path, lines)
        dontcare = parse_dontcare_regions(p)
        assert dontcare == {}

    def test_missing_file(self, tmp_path):
        dontcare = parse_dontcare_regions(tmp_path / "nonexistent.txt")
        assert dontcare == {}

    def test_multiple_dontcare_same_frame(self, tmp_path):
        lines = [
            "0 -1 DontCare -1 -1 -10 0 0 50 50 -1 -1 -1 -1 -1 -1 -1",
            "0 -1 DontCare -1 -1 -10 100 100 150 150 -1 -1 -1 -1 -1 -1 -1",
        ]
        p = self._write_label(tmp_path, lines)
        dontcare = parse_dontcare_regions(p)
        assert dontcare[0].shape == (2, 4)


class TestParseIgnoreRegions:
    """parse_ignore_regions generalizes DontCare to also catch Van/Truck/etc."""

    def _write_label(self, tmp_path: Path, lines: list) -> Path:
        p = tmp_path / "0000.txt"
        p.write_text("\n".join(lines))
        return p

    def test_van_is_ignored_not_evaluated(self, tmp_path):
        """A Van GT box should appear in ignore regions when allowed_classes
        is restricted to Car/Pedestrian/Cyclist — fixing the bug where a
        correctly-detected van became a guaranteed false positive."""
        lines = [
            "0 1 Car 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0",
            "0 2 Van 0.0 0 0.0 200 200 300 300 1 1 1 0 0 0 0",
        ]
        p = self._write_label(tmp_path, lines)
        ignore = parse_ignore_regions(p, allowed_classes=["Car", "Pedestrian", "Cyclist"])
        assert 0 in ignore
        assert ignore[0].shape == (1, 4)
        np.testing.assert_array_almost_equal(ignore[0][0], [200, 200, 300, 300])

    def test_dontcare_and_van_both_ignored(self, tmp_path):
        lines = [
            "0 -1 DontCare -1 -1 -10 0 0 50 50 -1 -1 -1 -1 -1 -1 -1",
            "0 2 Truck 0.0 0 0.0 200 200 300 300 1 1 1 0 0 0 0",
        ]
        p = self._write_label(tmp_path, lines)
        ignore = parse_ignore_regions(p, allowed_classes=["Car", "Pedestrian", "Cyclist"])
        assert ignore[0].shape == (2, 4)

    def test_car_not_in_ignore_regions(self, tmp_path):
        """Sanity check: a normal Car GT box should NOT show up as an
        ignore region — only non-evaluated classes should."""
        lines = ["0 1 Car 0.0 0 0.0 0 0 100 100 1 1 1 0 0 0 0"]
        p = self._write_label(tmp_path, lines)
        ignore = parse_ignore_regions(p, allowed_classes=["Car", "Pedestrian", "Cyclist"])
        assert ignore == {}


# ── MOTA Evaluator ────────────────────────────────────────────────────────────

def _make_box(frame, tid, cls="Car", x1=0, y1=0, x2=100, y2=100):
    return KITTIBox(
        frame=frame, track_id=tid, cls=cls,
        truncated=0.0, occluded=0, alpha=0.0,
        x1=x1, y1=y1, x2=x2, y2=y2,
    )


class TestMOTAEvaluator:
    def test_perfect_tracking(self):
        """One GT box perfectly matched → MOTA should be 100."""
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])
        for fid in range(5):
            gt   = [_make_box(fid, 1, "Car", 0, 0, 100, 100)]
            pred = np.array([[0, 0, 100, 100]], dtype=np.float32)
            ids  = np.array([42])
            cls  = np.array([0])
            evaluator.update(fid, gt, pred, ids, cls)

        df = evaluator.compute()
        # MOTA = 100 means zero FP, zero FN, zero IDS
        assert "All" in df.index
        assert float(df.loc["All", "mota"]) >= 99.0

    def test_no_detections(self):
        """All misses → MOTA should be ≤ 0."""
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])
        for fid in range(5):
            gt   = [_make_box(fid, 1, "Car")]
            pred = np.zeros((0, 4), dtype=np.float32)
            ids  = np.array([], dtype=int)
            cls  = np.array([], dtype=int)
            evaluator.update(fid, gt, pred, ids, cls)

        df = evaluator.compute()
        assert float(df.loc["All", "mota"]) <= 0.0

    def test_reset(self):
        """After reset, a fresh run returns results for the new data only."""
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])
        gt   = [_make_box(0, 1, "Car")]
        pred = np.array([[0, 0, 100, 100]], dtype=np.float32)
        ids  = np.array([1])
        cls  = np.array([0])
        evaluator.update(0, gt, pred, ids, cls)

        evaluator.reset()
        df = evaluator.compute()
        # After reset, no frames → accumulators are empty
        # motmetrics returns NaN for MOTA with zero events
        mota = df.loc["All", "mota"] if "All" in df.index else float("nan")
        # Accept NaN or 0 (both valid for zero-event accumulators)
        assert (mota != mota) or (mota == 0.0) or (abs(mota) < 1e-3)

    def test_compute_mota_scalar(self):
        """compute_mota() returns a float."""
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])
        gt   = [_make_box(0, 1, "Car")]
        pred = np.array([[0, 0, 100, 100]], dtype=np.float32)
        evaluator.update(0, gt, pred, np.array([1]), np.array([0]))
        mota = evaluator.compute_mota()
        assert isinstance(mota, float)

    def test_empty_frame(self):
        """Empty GT + empty pred should not crash."""
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])
        evaluator.update(
            frame_id=0,
            gt_boxes=[],
            pred_xyxy=np.zeros((0, 4), dtype=np.float32),
            pred_ids=np.array([], dtype=int),
            pred_classes=np.array([], dtype=int),
        )
        df = evaluator.compute()
        assert df is not None

    def test_dontcare_filtering_removes_false_positive(self):
        """
        A prediction overlapping a DontCare region should NOT be counted
        as a false positive — per KITTI's official evaluation protocol.
        """
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])

        # One real GT car, perfectly matched
        gt = [_make_box(0, 1, "Car", 0, 0, 100, 100)]

        # Two predictions: one matches GT, one is a "background" detection
        # that exactly overlaps a DontCare region (e.g. a distant unlabeled car)
        pred_xyxy = np.array([
            [0, 0, 100, 100],        # matches GT — true positive
            [490, 490, 550, 540],    # exactly overlaps DontCare — should be excluded
        ], dtype=np.float32)
        pred_ids = np.array([10, 11])
        pred_cls = np.array([0, 0])

        dontcare = np.array([[490, 490, 550, 540]], dtype=np.float32)

        evaluator.update(
            frame_id=0, gt_boxes=gt,
            pred_xyxy=pred_xyxy, pred_ids=pred_ids, pred_classes=pred_cls,
            dontcare_xyxy=dontcare,
        )

        df = evaluator.compute()
        # With DontCare filtering, FP should be 0 (only the matched box remains)
        assert df.loc["All", "num_false_positives"] == 0
        # MOTA should be high/perfect since the only "bad" prediction was excluded
        assert df.loc["All", "mota"] >= 99.0

    def test_dontcare_filtering_disabled_counts_fp(self):
        """
        Sanity check: WITHOUT dontcare_xyxy, the same background prediction
        SHOULD be counted as a false positive — confirms the filter is
        actually doing something, not just a no-op.
        """
        evaluator = MOTAEvaluator(iou_threshold=0.5, classes=["Car"])

        gt = [_make_box(0, 1, "Car", 0, 0, 100, 100)]
        pred_xyxy = np.array([
            [0, 0, 100, 100],
            [490, 490, 550, 540],
        ], dtype=np.float32)
        pred_ids = np.array([10, 11])
        pred_cls = np.array([0, 0])

        evaluator.update(
            frame_id=0, gt_boxes=gt,
            pred_xyxy=pred_xyxy, pred_ids=pred_ids, pred_classes=pred_cls,
            dontcare_xyxy=None,   # no filtering
        )

        df = evaluator.compute()
        assert df.loc["All", "num_false_positives"] == 1


# ── Visualizer smoke test ─────────────────────────────────────────────────────

class TestFrameAnnotator:
    def test_annotate_no_detections(self):
        """Annotating a frame with no detections should return an image."""
        import supervision as sv
        from src.visualizer import FrameAnnotator

        frame     = np.zeros((375, 1242, 3), dtype=np.uint8)
        annotator = FrameAnnotator()
        result    = annotator.annotate(
            frame,
            sv.Detections.empty(),
            histories={},
            frame_id=0,
            seq_name="0000",
        )
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_annotate_with_detections(self):
        import supervision as sv
        from src.visualizer import FrameAnnotator

        frame = np.zeros((375, 1242, 3), dtype=np.uint8)
        dets  = sv.Detections(
            xyxy=np.array([[100, 50, 300, 200]], dtype=np.float32),
            confidence=np.array([0.85]),
            class_id=np.array([0]),
            tracker_id=np.array([7]),
        )
        annotator = FrameAnnotator()
        result    = annotator.annotate(frame, dets, {}, frame_id=1, seq_name="0000")
        assert result.shape == frame.shape
        # Frame should have been modified (drawn on)
        assert not np.all(result == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
