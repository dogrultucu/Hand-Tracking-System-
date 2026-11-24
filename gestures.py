from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

from config import AppConfig
from primitives import PrimitiveStateRecord
from tracker import TrackedHand


@dataclass
class LowLevelActionDecision:
    hand_id: str
    handedness: str
    frame_index: int
    timestamp: float
    action_label: str
    action_confidence: float
    action_scores: Dict[str, float]
    primitive_states: List[str]
    dominant_features: Dict[str, float]
    debug_explanation: str

    def to_record(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "action_label": self.action_label,
            "action_confidence": self.action_confidence,
            "primitive_states": "|".join(self.primitive_states),
            "debug_explanation": self.debug_explanation,
        }


class LowLevelActionAnalyzer:
    """Infers per-frame low-level actions from primitive states and short temporal windows."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.history: Dict[str, Deque[Tuple[PrimitiveStateRecord, Dict[str, float]]]] = defaultdict(
            lambda: deque(maxlen=self.config.gestures.analysis_window)
        )

    def update(
        self,
        hand: TrackedHand,
        primitive: PrimitiveStateRecord,
        features: Dict[str, float],
        frame_index: int,
        timestamp: float,
    ) -> LowLevelActionDecision:
        history = self.history[hand.track_id]
        history.append((primitive, features))
        scores = self._score_actions(history)
        label, confidence = max(scores.items(), key=lambda item: item[1])
        if confidence < self.config.gestures.min_confidence_for_action:
            label = self.config.gestures.idle_label
            confidence = max(scores.get(self.config.gestures.idle_label, 0.0), 1.0 - confidence)
        primitive_states = [
            primitive.posture_state,
            primitive.transition_state,
            primitive.motion_state,
            primitive.rotation_state,
            primitive.direction_state,
        ]
        dominant_features = {
            "engaged_score": float(features["engaged_score"]),
            "motion_magnitude": float(features["motion_magnitude"]),
            "rotation_velocity": float(features["rotation_velocity"]),
            "openness_score": float(features["openness_score"]),
        }
        return LowLevelActionDecision(
            hand_id=hand.track_id,
            handedness=hand.handedness,
            frame_index=frame_index,
            timestamp=timestamp,
            action_label=label,
            action_confidence=float(confidence),
            action_scores=scores,
            primitive_states=primitive_states,
            dominant_features=dominant_features,
            debug_explanation=self._build_explanation(label, primitive, features, history),
        )

    def _score_actions(self, history: Deque[Tuple[PrimitiveStateRecord, Dict[str, float]]]) -> Dict[str, float]:
        primitive, features = history[-1]
        primitives = [item[0] for item in history]
        feature_window = [item[1] for item in history]
        engaged = float(features["engaged_score"])
        motion = float(features["motion_magnitude"])
        posture_stability = float(np.clip(1.0 - features["posture_change"], 0.0, 1.0))
        visibility = float(features["visibility_score"])
        side_view_score = float(features.get("side_view_score", 0.0))
        vertical_motion_strength = float(np.clip(abs(float(features["motion_dy"])) / 0.12, 0.0, 1.5))

        closed_now = primitive.posture_state == "closed"
        moving_now = primitive.motion_state == "moving"
        stable_now = primitive.motion_state == "stable"
        rotating_now = primitive.rotation_state in {"rotating", "rotating_cw", "rotating_ccw"}
        forward_now = primitive.direction_state == "forward_motion"
        backward_now = primitive.direction_state == "backward_motion"
        opening_now = primitive.transition_state == "opening"
        closing_now = primitive.transition_state == "closing"

        recent_postures = [item.posture_state for item in primitives]
        recent_transitions = [item.transition_state for item in primitives]
        had_open_recently = "open" in recent_postures[:-1]
        had_closed_recently = "closed" in recent_postures[:-1]
        had_closing_recently = "closing" in recent_transitions
        mean_rotation = float(np.mean([abs(item.rotation_score) for item in primitives]))
        mean_motion = float(np.mean([item.motion_magnitude for item in primitives]))
        recent_forward = sum(1 for item in primitives if item.direction_state == "forward_motion")
        recent_backward = sum(1 for item in primitives if item.direction_state == "backward_motion")
        directional_ratio = max(recent_forward, recent_backward) / max(len(primitives), 1)
        side_view_directional = side_view_score >= 0.55 and (forward_now or backward_now)
        directional_window_floor = 0.8 if side_view_directional else 0.0
        push_pull_activation = max(engaged, 0.9 if side_view_directional else 0.45)

        scores = {
            self.config.gestures.idle_label: float(np.clip(1.0 - max(engaged, mean_motion * 0.25), 0.0, 1.0)),
            "grab": 0.0,
            "hold": 0.0,
            "turn": 0.0,
            "pull": 0.0,
            "push": 0.0,
            "pull + turn": 0.0,
            "push + turn": 0.0,
            "release + push": 0.0,
            "move + hold": 0.0,
        }

        grab_transition = 1.0 if (closed_now and had_open_recently and (closing_now or had_closing_recently)) else 0.0
        scores["grab"] = float(np.clip(grab_transition * engaged * visibility, 0.0, 1.0))

        scores["hold"] = float(
            np.clip(
                float(closed_now) * float(stable_now) * max(engaged, 0.6) * posture_stability * visibility,
                0.0,
                1.0,
            )
        )
        scores["turn"] = float(np.clip(float(rotating_now) * max(engaged, 0.45) * mean_rotation * visibility, 0.0, 1.0))
        scores["push"] = float(
            np.clip(
                float(forward_now)
                * push_pull_activation
                * max((recent_forward / max(len(primitives), 1)), directional_window_floor)
                * max(1.0, vertical_motion_strength if side_view_score >= 0.55 else 1.0)
                * visibility,
                0.0,
                1.0,
            )
        )
        scores["pull"] = float(
            np.clip(
                float(backward_now)
                * push_pull_activation
                * max((recent_backward / max(len(primitives), 1)), directional_window_floor)
                * max(1.0, vertical_motion_strength if side_view_score >= 0.55 else 1.0)
                * visibility,
                0.0,
                1.0,
            )
        )
        scores["push + turn"] = float(np.clip(min(scores["push"], scores["turn"]) * 1.08, 0.0, 1.0))
        scores["pull + turn"] = float(np.clip(min(scores["pull"], scores["turn"]) * 1.08, 0.0, 1.0))
        scores["release + push"] = float(
            np.clip(
                float(opening_now or had_closed_recently)
                * float(forward_now)
                * max(features["openness_score"] - 0.4, 0.0)
                * visibility,
                0.0,
                1.0,
            )
        )
        scores["move + hold"] = float(
            np.clip(
                float(closed_now or engaged >= self.config.gestures.engaged_score_threshold)
                * float(moving_now)
                * posture_stability
                * visibility,
                0.0,
                1.0,
            )
        )

        if motion < self.config.gestures.hold_motion_threshold:
            scores["push"] *= 0.65
            scores["pull"] *= 0.65
            scores["push + turn"] *= 0.7
            scores["pull + turn"] *= 0.7
            scores["move + hold"] *= 0.75
        if mean_rotation < self.config.gestures.turn_rate_threshold * 0.6:
            scores["turn"] *= 0.7
            scores["push + turn"] *= 0.8
            scores["pull + turn"] *= 0.8
        if side_view_directional:
            scores["turn"] *= 0.35
        if directional_ratio >= 0.55 and side_view_score >= 0.55:
            scores["turn"] *= 0.35
        if engaged < self.config.gestures.engaged_score_threshold:
            for label in ("hold", "turn", "push + turn", "pull + turn", "move + hold"):
                scores[label] *= 0.55
            if not side_view_directional:
                scores["push"] *= 0.55
                scores["pull"] *= 0.55

        return scores

    def _build_explanation(
        self,
        label: str,
        primitive: PrimitiveStateRecord,
        features: Dict[str, float],
        history: Deque[Tuple[PrimitiveStateRecord, Dict[str, float]]],
    ) -> str:
        if label == self.config.gestures.idle_label:
            return "idle selected because no action score exceeded the minimum action confidence threshold"
        if label == "grab":
            return (
                "grab detected because openness decreased, closing persisted, "
                "and closed posture stabilized after an open state"
            )
        if label == "hold":
            return "hold detected because closed posture and stable motion remained active across the recent window"
        if label == "turn":
            return "turn detected because engaged posture overlapped with sustained rotating primitive states"
        if label == "push":
            return "push detected because engaged posture overlapped with forward_motion in the configured push axis"
        if label == "pull":
            return "pull detected because engaged posture overlapped with backward_motion in the configured pull axis"
        if label == "push + turn":
            return "push + turn detected because forward_motion and rotating primitives overlapped above thresholds"
        if label == "pull + turn":
            return "pull + turn detected because backward_motion and rotating primitives overlapped above thresholds"
        if label == "release + push":
            return "release + push detected because opening followed a previously closed state and forward motion remained active"
        if label == "move + hold":
            return "move + hold detected because engaged hold-like posture persisted while translational movement continued"
        return (
            f"{label} detected from primitive states "
            f"{primitive.posture_state}, {primitive.transition_state}, {primitive.motion_state}, "
            f"{primitive.rotation_state}, {primitive.direction_state}"
        )


GestureAnalyzer = LowLevelActionAnalyzer
