from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

pytest.importorskip("pydantic")

from backend.app.models.schemas import BoundingBox, DetectionBox
from backend.app.safety.analyzer import SafetyAnalyzer


def _make_detection(label: str, confidence: float = 0.9,
                    xmin: float = 10, ymin: float = 10,
                    xmax: float = 110, ymax: float = 210) -> DetectionBox:
    return DetectionBox(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, normalized=False,
        ),
        class_id=0,
    )


class TestSafetyAnalyzerEmptyDetections:
    def test_empty_detections_returns_empty_alerts(self):
        """SafetyAnalyzer with empty detections returns empty alerts."""
        analyzer = SafetyAnalyzer(temporal_threshold=1)
        alerts = analyzer.analyze_frame(detections=[], temporal_history=False)
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_none_detections_treated_as_empty(self):
        """Passing None detections should behave the same as empty list."""
        analyzer = SafetyAnalyzer(temporal_threshold=1)
        alerts = analyzer.analyze_frame(detections=None, temporal_history=False)
        assert alerts == []


class TestSafetyAnalyzerPersonDetection:
    def test_person_detection_returns_person_alert(self):
        """SafetyAnalyzer with Person detection returns person alert."""
        analyzer = SafetyAnalyzer(temporal_threshold=1)
        person = _make_detection("person", confidence=0.95)
        alerts = analyzer.analyze_frame(
            detections=[person],
            temporal_history=False,
            frame_shape=(480, 640),
        )
        person_alerts = [a for a in alerts if a.category == "person"]
        assert len(person_alerts) > 0
        a = person_alerts[0]
        assert a.severity == "info"
        assert "person" in a.message.lower()
        assert a.related_label == "person"


class TestSafetyAnalyzerFallDetection:
    def test_fall_requires_temporal_history_3_frames(self):
        """Fall detection should only fire after 3+ frames of confirmation."""
        analyzer = SafetyAnalyzer(
            temporal_threshold=3,
            fall_aspect_ratio_threshold=1.35,
            fall_aspect_delta_threshold=0.05,
            history_frames=10,
            obstacle_proximity_threshold=0.99,
        )

        standing = _make_detection(
            "person", confidence=0.9,
            xmin=100, ymin=50, xmax=200, ymax=450,
        )

        first_alerts = analyzer.analyze_frame(
            detections=[standing],
            temporal_history=True,
            frame_shape=(480, 640),
        )
        fall_first = [a for a in first_alerts if a.category == "fall"]
        assert len(fall_first) == 0, "Fall should not trigger on first frame"

        analyzer.analyze_frame(
            detections=[standing],
            temporal_history=True,
            frame_shape=(480, 640),
        )

        fallen = _make_detection(
            "person", confidence=0.9,
            xmin=50, ymin=250, xmax=550, ymax=400,
        )

        third_alerts = analyzer.analyze_frame(
            detections=[fallen],
            temporal_history=True,
            frame_shape=(480, 640),
        )
        fall_third = [a for a in third_alerts if a.category == "fall"]
        assert len(fall_third) == 0, "Fall still needs 3 confirmations of the fallen state"

        analyzer.analyze_frame(
            detections=[fallen],
            temporal_history=True,
            frame_shape=(480, 640),
        )

        fifth_alerts = analyzer.analyze_frame(
            detections=[fallen],
            temporal_history=True,
            frame_shape=(480, 640),
        )
        fall_fifth = [a for a in fifth_alerts if a.category == "fall"]
        assert len(fall_fifth) > 0, (
            "Fall should trigger after enough consecutive frames with flat aspect "
            f"ratio history. Alerts: {[(a.category, a.message) for a in fifth_alerts]}"
        )
