from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np

from config import AppConfig
from tracker import TrackedHand


@dataclass
class PrimitiveStateRecord:
    frame_index: int
    timestamp: float
    hand_id: str
    handedness: str
    openness: float
    motion_magnitude: float
    rotation_score: float
    primitive_labels: List[str]
    primitive_confidences: Dict[str, float]
    posture_state: str
    transition_state: str
    motion_state: str
    rotation_state: str
    direction_state: str

    def to_record(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "openness": self.openness,
            "motion_magnitude": self.motion_magnitude,
            "rotation_score": self.rotation_score,
            "posture_state": self.posture_state,
            "transition_state": self.transition_state,
            "motion_state": self.motion_state,
            "rotation_state": self.rotation_state,
            "direction_state": self.direction_state,
            "primitive_labels": "|".join(self.primitive_labels),
            "primitive_confidences": json.dumps(self.primitive_confidences, sort_keys=True),
        }


@dataclass
class _CategoryState:
    stable_label: str = "unknown"
    pending_label: str | None = None
    pending_count: int = 0


@dataclass
class _PrimitiveMemory:
    categories: Dict[str, _CategoryState] = field(default_factory=dict)
    history: Deque[PrimitiveStateRecord] = field(default_factory=lambda: deque(maxlen=16))


class PrimitiveStateAnalyzer:
    """Converts hand features into confidence-aware primitive motion states."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.memories: Dict[str, _PrimitiveMemory] = defaultdict(_PrimitiveMemory)
        self.push_axis = self._normalize_axis(np.array(config.gestures.push_axis, dtype=float))

    def update(
        self,
        hand: TrackedHand,
        features: Dict[str, float],
        frame_index: int,
        timestamp: float,
    ) -> PrimitiveStateRecord:
        memory = self.memories[hand.track_id]
        visibility = float(features["visibility_score"])
        openness = float(features["openness_score"])
        motion = float(features["motion_magnitude"])
        rotation_velocity = float(features["rotation_velocity"])
        motion_vec = np.array([features["motion_dx"], features["motion_dy"], features["motion_dz"]], dtype=float)
        push_projection = float(np.dot(motion_vec, self.push_axis))
        side_view_score = float(features.get("side_view_score", 0.0))
        abs_dx = abs(float(features["motion_dx"]))
        abs_dy = abs(float(features["motion_dy"]))
        abs_dz = abs(float(features["motion_dz"]))
        previous = memory.history[-1] if memory.history else None
        openness_delta = 0.0 if previous is None else openness - previous.openness

        posture_label = self._posture_label(openness, visibility)
        posture_confidence = self._posture_confidence(openness, posture_label, visibility)

        transition_label = "neutral"
        transition_confidence = 0.0
        if visibility < self.config.primitives.uncertain_visibility_threshold:
            transition_label = "uncertain_motion"
            transition_confidence = 1.0 - visibility
        elif openness_delta >= self.config.primitives.opening_delta_threshold:
            transition_label = "opening"
            transition_confidence = min(1.0, openness_delta / self.config.primitives.opening_delta_threshold)
        elif openness_delta <= -self.config.primitives.closing_delta_threshold:
            transition_label = "closing"
            transition_confidence = min(1.0, abs(openness_delta) / self.config.primitives.closing_delta_threshold)

        motion_label = "stable"
        if visibility < self.config.primitives.uncertain_visibility_threshold:
            motion_label = "uncertain_motion"
        elif motion >= self.config.primitives.moving_motion_threshold:
            motion_label = "moving"
        elif motion > self.config.primitives.stable_motion_threshold:
            motion_label = "uncertain_motion"
        motion_confidence = self._motion_confidence(motion, motion_label, visibility)

        rotation_label = "stable"
        if visibility < self.config.primitives.uncertain_visibility_threshold:
            rotation_label = "uncertain_motion"
        elif abs(rotation_velocity) >= self.config.primitives.rotating_threshold:
            rotation_label = "rotating"
            if rotation_velocity >= self.config.primitives.rotation_direction_threshold:
                rotation_label = "rotating_ccw"
            elif rotation_velocity <= -self.config.primitives.rotation_direction_threshold:
                rotation_label = "rotating_cw"
        rotation_confidence = self._rotation_confidence(rotation_velocity, visibility)

        direction_label = "stable"
        if visibility < self.config.primitives.uncertain_visibility_threshold:
            direction_label = "uncertain_motion"
        elif self._is_side_view_direction_motion(side_view_score, abs_dx, abs_dy, abs_dz):
            direction_label = "forward_motion" if float(features["motion_dy"]) >= 0.0 else "backward_motion"
        elif push_projection >= 0.24:
            direction_label = "forward_motion"
        elif push_projection <= -0.24:
            direction_label = "backward_motion"
        elif motion_label == "moving":
            direction_label = "uncertain_motion"
        direction_confidence = self._direction_confidence(push_projection, visibility)
        if self._is_side_view_direction_motion(side_view_score, abs_dx, abs_dy, abs_dz):
            direction_confidence = max(direction_confidence, float(np.clip((abs_dy / max(self.config.primitives.side_view_vertical_motion_threshold, 1e-6)) * visibility, 0.0, 1.0)))

        stabilized = {
            "posture": self._stabilize(hand.track_id, "posture", posture_label),
            "transition": self._stabilize(hand.track_id, "transition", transition_label),
            "motion": self._stabilize(hand.track_id, "motion", motion_label),
            "rotation": self._stabilize(hand.track_id, "rotation", rotation_label),
            "direction": self._stabilize(hand.track_id, "direction", direction_label),
        }

        primitive_confidences = {
            f"posture:{stabilized['posture']}": posture_confidence,
            f"transition:{stabilized['transition']}": transition_confidence,
            f"motion:{stabilized['motion']}": motion_confidence,
            f"rotation:{stabilized['rotation']}": rotation_confidence,
            f"direction:{stabilized['direction']}": direction_confidence,
        }
        primitive_labels = [
            label
            for label in [
                stabilized["posture"],
                stabilized["transition"],
                stabilized["motion"],
                stabilized["rotation"],
                stabilized["direction"],
            ]
            if label not in {"unknown", "neutral"}
        ]

        record = PrimitiveStateRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            hand_id=hand.track_id,
            handedness=hand.handedness,
            openness=openness,
            motion_magnitude=motion,
            rotation_score=abs(rotation_velocity),
            primitive_labels=primitive_labels,
            primitive_confidences=primitive_confidences,
            posture_state=stabilized["posture"],
            transition_state=stabilized["transition"],
            motion_state=stabilized["motion"],
            rotation_state=stabilized["rotation"],
            direction_state=stabilized["direction"],
        )
        memory.history.append(record)
        return record

    def _stabilize(self, hand_id: str, category: str, raw_label: str) -> str:
        memory = self.memories[hand_id]
        category_state = memory.categories.setdefault(category, _CategoryState())
        if raw_label == category_state.stable_label:
            category_state.pending_label = None
            category_state.pending_count = 0
            return category_state.stable_label

        if category_state.pending_label == raw_label:
            category_state.pending_count += 1
        else:
            category_state.pending_label = raw_label
            category_state.pending_count = 1

        if category_state.pending_count >= self.config.primitives.primitive_confirm_frames:
            category_state.stable_label = raw_label
            category_state.pending_label = None
            category_state.pending_count = 0
        return category_state.stable_label

    def _posture_label(self, openness: float, visibility: float) -> str:
        if visibility < self.config.primitives.uncertain_visibility_threshold:
            return "unknown"
        if openness >= self.config.primitives.open_threshold:
            return "open"
        if openness <= self.config.primitives.closed_threshold:
            return "closed"
        return "unknown"

    def _posture_confidence(self, openness: float, label: str, visibility: float) -> float:
        if label == "open":
            margin = openness - self.config.primitives.open_threshold
            return float(np.clip(visibility * (0.65 + margin), 0.0, 1.0))
        if label == "closed":
            margin = self.config.primitives.closed_threshold - openness
            return float(np.clip(visibility * (0.65 + margin), 0.0, 1.0))
        return float(np.clip(1.0 - abs(openness - 0.5), 0.0, 1.0) * visibility * 0.35)

    def _motion_confidence(self, motion: float, label: str, visibility: float) -> float:
        if label == "moving":
            return float(np.clip((motion / self.config.primitives.moving_motion_threshold) * visibility, 0.0, 1.0))
        if label == "stable":
            return float(np.clip((1.0 - (motion / max(self.config.primitives.moving_motion_threshold, 1e-6))) * visibility, 0.0, 1.0))
        return float(np.clip(visibility * 0.45, 0.0, 1.0))

    def _rotation_confidence(self, rotation_velocity: float, visibility: float) -> float:
        return float(
            np.clip((abs(rotation_velocity) / max(self.config.primitives.rotating_threshold, 1e-6)) * visibility, 0.0, 1.0)
        )

    def _direction_confidence(self, push_projection: float, visibility: float) -> float:
        return float(np.clip((abs(push_projection) / 0.24) * visibility, 0.0, 1.0))

    def _is_side_view_direction_motion(
        self,
        side_view_score: float,
        abs_dx: float,
        abs_dy: float,
        abs_dz: float,
    ) -> bool:
        if side_view_score < self.config.primitives.side_view_direction_min_score:
            return False
        if abs_dy < self.config.primitives.side_view_vertical_motion_threshold:
            return False
        dominant_other = max(abs_dx, abs_dz)
        return abs_dy >= (dominant_other * self.config.primitives.side_view_vertical_dominance_ratio)

    @staticmethod
    def _normalize_axis(axis: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=float)
        return axis / norm
