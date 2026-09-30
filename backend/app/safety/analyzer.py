from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

from ..models.schemas import BoundingBox, DetectionBox, SafetyAlert
from ..utils.config import get_settings
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


Severity = Literal["info", "warning", "critical"]


@dataclass
class _TemporalEntry:
    """Internal record for a single detection we track over time."""

    key: str
    label: str
    first_seen: float
    last_seen: float
    last_aspect: float
    aspect_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    confirmations: int = 0


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax, ay = max(a.xmin, b.xmin), max(a.ymin, b.ymin)
    bx, by = min(a.xmax, b.xmax), min(a.ymax, b.ymax)
    inter_w, inter_h = max(0.0, bx - ax), max(0.0, by - ay)
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _proximity(a: BoundingBox, b: BoundingBox) -> float:
    """
    Normalised proximity in 0..1 between two boxes (1 = perfectly overlapping).

    Uses centre-point distance normalised by the sum of their diagonal half
    lengths, clamped to [0, 1].
    """
    axc = (a.xmin + a.xmax) / 2.0
    ayc = (a.ymin + a.ymax) / 2.0
    bxc = (b.xmin + b.xmax) / 2.0
    byc = (b.ymin + b.ymax) / 2.0
    dist = ((axc - bxc) ** 2 + (ayc - byc) ** 2) ** 0.5
    diag_a = (a.width ** 2 + a.height ** 2) ** 0.5 / 2.0
    diag_b = (b.width ** 2 + b.height ** 2) ** 0.5 / 2.0
    norm = diag_a + diag_b + 1e-6
    p = 1.0 - min(1.0, dist / norm)
    return max(0.0, min(1.0, p))


class SafetyAnalyzer:
    """
    Rule-based safety analyser operating on detection boxes.

    Categories of alerts produced:
      * ``person``      - any person detected (INFO severity)
      * ``obstacle``    - person close to another object (WARNING+)
      * ``fall``        - likely fall based on person bbox aspect ratio shift
      * ``smoke``/``fire`` - placeholder stub if a fire/smoke label appears
      * ``other``       - misc

    Temporal behaviour:
      * The last ``history_frames`` worth of detections are retained internally.
      * An alert requires at least ``temporal_threshold`` consecutive hits
        before it is raised (to avoid spurious single-frame glitches).
      * ``analyze_frame`` is fully deterministic given identical history.

    Never contacts the network.
    """

    def __init__(
        self,
        obstacle_proximity_threshold: float = 0.55,
        fall_aspect_ratio_threshold: float = 1.35,
        fall_aspect_delta_threshold: float = 0.45,
        temporal_threshold: int = 3,
        history_frames: int = 60,
        smoke_fire_labels: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._obstacle_p_thresh = float(obstacle_proximity_threshold)
        self._fall_ar_thresh = float(fall_aspect_ratio_threshold)
        self._fall_ar_delta = float(fall_aspect_delta_threshold)
        self._temporal_thresh = max(1, int(temporal_threshold))
        self._history_frames = max(1, int(history_frames))
        self._smoke_fire_labels = tuple(
            smoke_fire_labels or ("smoke", "fire", "flame")
        )
        self._temporal: Dict[str, _TemporalEntry] = {}
        self._frame_history: Deque[List[DetectionBox]] = deque(maxlen=self._history_frames)
        self._last_alerts: List[SafetyAlert] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _track_key(label: str, bbox: BoundingBox, frame_h: int, frame_w: int) -> str:
        cx = int((bbox.xmin + bbox.xmax) / 2.0 / max(1, frame_w) * 32)
        cy = int((bbox.ymin + bbox.ymax) / 2.0 / max(1, frame_h) * 32)
        return f"{label.lower()}::{cx}x{cy}"

    def _get_or_create_entry(self, key: str, label: str, aspect: float) -> _TemporalEntry:
        now = time.perf_counter()
        entry = self._temporal.get(key)
        if entry is None:
            entry = _TemporalEntry(
                key=key,
                label=label,
                first_seen=now,
                last_seen=now,
                last_aspect=aspect,
            )
            self._temporal[key] = entry
        else:
            entry.last_seen = now
        entry.aspect_samples.append(aspect)
        entry.last_aspect = aspect
        return entry

    def _expire_old_entries(self, ttl_seconds: float = 5.0) -> None:
        now = time.perf_counter()
        stale = [k for k, v in self._temporal.items() if (now - v.last_seen) > ttl_seconds]
        for k in stale:
            self._temporal.pop(k, None)

    # ------------------------------------------------------------------
    # Category checks
    # ------------------------------------------------------------------
    def _check_person(self, dets: List[DetectionBox]) -> List[SafetyAlert]:
        alerts: List[SafetyAlert] = []
        people = [d for d in dets if d.label.lower() == "person"]
        for p in people:
            alert = SafetyAlert(
                severity="info",
                category="person",
                message="Person detected in the scene.",
                confidence=round(p.confidence, 3),
                related_label=p.label,
                bbox=p.bbox,
                requires_confirmation=False,
                temporal_confirmations=1,
            )
            alerts.append(alert)
        return alerts

    def _check_obstacle(self, dets: List[DetectionBox]) -> List[SafetyAlert]:
        alerts: List[SafetyAlert] = []
        people = [d for d in dets if d.label.lower() == "person"]
        others = [d for d in dets if d.label.lower() != "person"]
        if not people or not others:
            return alerts
        for person in people:
            closest: Optional[DetectionBox] = None
            closest_p = 0.0
            for other in others:
                prox = _proximity(person.bbox, other.bbox)
                if prox > closest_p:
                    closest_p = prox
                    closest = other
            if closest is not None and closest_p >= self._obstacle_p_thresh:
                sev: Severity = "critical" if closest_p >= 0.85 else "warning"
                conf = round(min(1.0, closest_p), 3)
                alerts.append(
                    SafetyAlert(
                        severity=sev,
                        category="obstacle",
                        message=(
                            f"Obstacle detected: person is close to '{closest.label}'. "
                            "Please confirm and move carefully."
                        ),
                        confidence=conf,
                        related_label=closest.label,
                        bbox=closest.bbox,
                        requires_confirmation=True,
                        temporal_confirmations=1,
                    )
                )
        return alerts

    def _check_fall(
        self, dets: List[DetectionBox], frame_h: int, frame_w: int
    ) -> List[SafetyAlert]:
        alerts: List[SafetyAlert] = []
        people = [d for d in dets if d.label.lower() == "person"]
        if not people:
            return alerts
        for p in people:
            key = self._track_key(p.label, p.bbox, frame_h, frame_w)
            aspect = p.bbox.aspect_ratio
            entry = self._get_or_create_entry(key, p.label, aspect)
            entry.confirmations += 1
            if len(entry.aspect_samples) < 4:
                continue
            baseline = sum(list(entry.aspect_samples)[: -len(entry.aspect_samples) // 2 or 1]) / max(
                1, len(entry.aspect_samples) // 2 or 1
            )
            recent = sum(list(entry.aspect_samples)[-3:]) / min(3, len(entry.aspect_samples))
            delta = abs(recent - baseline)
            flat_lying = recent >= self._fall_ar_thresh
            sharp_change = delta >= self._fall_ar_delta
            if flat_lying or sharp_change:
                sev: Severity = "critical" if (flat_lying and sharp_change) else "warning"
                conf = round(min(1.0, 0.6 + (delta / 2.0) + (0.3 if flat_lying else 0.0)), 3)
                alerts.append(
                    SafetyAlert(
                        severity=sev,
                        category="fall",
                        message="Possible fall detected. Please confirm immediately.",
                        confidence=conf,
                        related_label=p.label,
                        bbox=p.bbox,
                        requires_confirmation=True,
                        temporal_confirmations=max(1, entry.confirmations),
                    )
                )
        return alerts

    def _check_smoke_fire(self, dets: List[DetectionBox]) -> List[SafetyAlert]:
        alerts: List[SafetyAlert] = []
        labels_lower = tuple(s.lower() for s in self._smoke_fire_labels)
        for d in dets:
            if d.label.lower() in labels_lower:
                alerts.append(
                    SafetyAlert(
                        severity="critical",
                        category="smoke" if "smoke" in d.label.lower() else "fire",
                        message=f"{d.label.capitalize()} detected. Evacuate and confirm immediately.",
                        confidence=round(d.confidence, 3),
                        related_label=d.label,
                        bbox=d.bbox,
                        requires_confirmation=True,
                        temporal_confirmations=1,
                    )
                )
        return alerts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_frame(
        self,
        detections: Optional[List[DetectionBox]] = None,
        temporal_history: bool = True,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[SafetyAlert]:
        """
        Run all safety checks against a list of detections.

        Args:
            detections: Detections from the latest frame.
            temporal_history: If False, ignore history state and run a
                single-shot analysis.
            frame_shape: Optional ``(H, W)`` of the source frame, used for
                normalisation. Auto-inferred from bounding-box ranges if
                omitted.
        """
        detections = list(detections or [])
        with self._lock:
            self._expire_old_entries()

            if frame_shape is not None:
                frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
            else:
                if detections:
                    frame_w = int(max(max(d.bbox.xmax for d in detections), 1.0))
                    frame_h = int(max(max(d.bbox.ymax for d in detections), 1.0))
                else:
                    frame_h, frame_w = 480, 640

            if temporal_history:
                self._frame_history.append(detections)

            alerts: List[SafetyAlert] = []
            alerts.extend(self._check_person(detections))
            alerts.extend(self._check_obstacle(detections))
            alerts.extend(self._check_fall(detections, frame_h, frame_w))
            alerts.extend(self._check_smoke_fire(detections))

            # Apply temporal confirmation threshold per category/key
            filtered: List[SafetyAlert] = []
            counts: Dict[str, int] = {}
            for alert in alerts:
                k = f"{alert.category}:{alert.related_label or ''}"
                counts[k] = counts.get(k, 0) + 1
                if alert.category in {"fall", "obstacle"} and temporal_history:
                    confirmations = sum(
                        1
                        for frame in list(self._frame_history)[-self._temporal_thresh:]
                        for d in frame
                        if alert.related_label and d.label == alert.related_label
                    )
                    if confirmations < self._temporal_thresh and alert.requires_confirmation:
                        continue
                filtered.append(alert)

            # Sort critical first
            severity_order = {"critical": 0, "warning": 1, "info": 2}
            filtered.sort(key=lambda a: (severity_order[a.severity], -a.confidence))
            self._last_alerts = filtered
            return filtered

    def last_alerts(self) -> List[SafetyAlert]:
        with self._lock:
            return list(self._last_alerts)

    def overall_severity(self, alerts: Optional[List[SafetyAlert]] = None) -> Severity:
        alerts = alerts if alerts is not None else self._last_alerts
        if not alerts:
            return "info"
        has_critical = any(a.severity == "critical" for a in alerts)
        has_warning = any(a.severity == "warning" for a in alerts)
        return "critical" if has_critical else ("warning" if has_warning else "info")

    def summary(self, alerts: Optional[List[SafetyAlert]] = None) -> str:
        alerts = alerts if alerts is not None else self._last_alerts
        if not alerts:
            return "No safety alerts in the latest frame."
        by_category: Dict[str, int] = {}
        for a in alerts:
            by_category[a.category] = by_category.get(a.category, 0) + 1
        parts = [f"{count} {name}" for name, count in sorted(by_category.items())]
        sev = self.overall_severity(alerts)
        return f"Overall {sev.upper()}: " + ", ".join(parts) + "."

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            alerts = list(self._last_alerts)
            return {
                "alerts": [a.model_dump() for a in alerts],
                "history_depth": len(self._frame_history),
                "tracked_objects": len(self._temporal),
                "overall_severity": self.overall_severity(alerts),
                "summary": self.summary(alerts),
            }


__all__ = ["SafetyAnalyzer"]
