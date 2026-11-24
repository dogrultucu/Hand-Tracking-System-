from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from config import AppConfig


@dataclass
class RelationFrameRecord:
    frame_index: int
    timestamp: float
    relation_label: str
    relation_confidence: float
    hand_ids: List[str]
    debug_explanation: str

    def to_record(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "relation_label": self.relation_label,
            "relation_confidence": self.relation_confidence,
            "hand_ids": "|".join(self.hand_ids),
            "debug_explanation": self.debug_explanation,
        }


class InterHandRelationAnalyzer:
    """Infers relation states between left/right hands from already-computed hand states."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.previous_distance: float | None = None

    def update(self, frame_index: int, timestamp: float, hand_entries: List[Dict[str, object]]) -> RelationFrameRecord:
        if len(hand_entries) < 2:
            self.previous_distance = None
            return RelationFrameRecord(
                frame_index=frame_index,
                timestamp=timestamp,
                relation_label=self.config.relations.idle_label,
                relation_confidence=0.35,
                hand_ids=[str(entry["hand_id"]) for entry in hand_entries],
                debug_explanation="independent assumed because fewer than two hands were active in this frame",
            )

        first = hand_entries[0]
        second = hand_entries[1]
        center_a = np.array([first["palm_center_x"], first["palm_center_y"]], dtype=float)
        center_b = np.array([second["palm_center_x"], second["palm_center_y"]], dtype=float)
        distance = float(np.linalg.norm(center_a - center_b) / max((first["hand_scale"] + second["hand_scale"]) * 0.5, 1e-6))
        distance_delta = 0.0 if self.previous_distance is None else distance - self.previous_distance
        self.previous_distance = distance

        motion_a = np.array([first["motion_dx"], first["motion_dy"]], dtype=float)
        motion_b = np.array([second["motion_dx"], second["motion_dy"]], dtype=float)
        motion_mag_a = float(np.linalg.norm(motion_a))
        motion_mag_b = float(np.linalg.norm(motion_b))
        cosine = self._cosine_similarity(motion_a, motion_b)
        opposite_cosine = self._cosine_similarity(motion_a, -motion_b)

        action_a = str(first["action_label"])
        action_b = str(second["action_label"])
        engaged_a = float(first["engaged_score"])
        engaged_b = float(second["engaged_score"])
        stable_a = str(first["primitive_motion_state"]) == "stable"
        stable_b = str(second["primitive_motion_state"]) == "stable"
        closed_a = str(first["primitive_posture_state"]) == "closed"
        closed_b = str(second["primitive_posture_state"]) == "closed"

        scores = {
            "independent": 0.45,
            "approaching": 0.0,
            "separating": 0.0,
            "coordinated": 0.0,
            "mirrored_motion": 0.0,
            "supporting": 0.0,
            "dual_hold": 0.0,
            "handoff_like": 0.0,
        }

        scores["approaching"] = float(
            np.clip(
                (-distance_delta / max(self.config.relations.approaching_speed_threshold, 1e-6))
                * min(motion_mag_a + motion_mag_b, 1.0),
                0.0,
                1.0,
            )
        )
        scores["separating"] = float(
            np.clip(
                (distance_delta / max(self.config.relations.separating_speed_threshold, 1e-6))
                * min(motion_mag_a + motion_mag_b, 1.0),
                0.0,
                1.0,
            )
        )
        scores["coordinated"] = float(
            np.clip(
                max(
                    0.0,
                    (cosine - self.config.relations.coordinated_similarity_threshold)
                    / max(1.0 - self.config.relations.coordinated_similarity_threshold, 1e-6),
                )
                * min(motion_mag_a, motion_mag_b)
                * max(engaged_a, engaged_b),
                0.0,
                1.0,
            )
        )
        scores["mirrored_motion"] = float(
            np.clip(
                max(
                    0.0,
                    (opposite_cosine - abs(self.config.relations.mirrored_similarity_threshold))
                    / max(1.0 - abs(self.config.relations.mirrored_similarity_threshold), 1e-6),
                )
                * min(motion_mag_a, motion_mag_b),
                0.0,
                1.0,
            )
        )
        scores["supporting"] = float(
            np.clip(
                (
                    (float(stable_a) * max(engaged_a, 0.4) * float(action_b != self.config.gestures.idle_label))
                    + (float(stable_b) * max(engaged_b, 0.4) * float(action_a != self.config.gestures.idle_label))
                )
                * self.config.relations.support_motion_threshold,
                0.0,
                1.0,
            )
        )
        scores["dual_hold"] = float(
            np.clip(
                float(closed_a and closed_b)
                * float(stable_a and stable_b)
                * min(engaged_a, engaged_b)
                / max(self.config.relations.dual_hold_min_confidence, 1e-6),
                0.0,
                1.0,
            )
        )
        scores["handoff_like"] = float(
            np.clip(
                (
                    float(action_a == "release + push" and action_b == "grab")
                    + float(action_b == "release + push" and action_a == "grab")
                )
                * max(0.0, 1.0 - (distance / max(self.config.relations.handoff_distance_delta_threshold, 1e-6)))
                * max(engaged_a, engaged_b),
                0.0,
                1.0,
            )
        )

        label, confidence = max(scores.items(), key=lambda item: item[1])
        if confidence < self.config.relations.relation_confidence_threshold:
            label = self.config.relations.idle_label
            confidence = max(scores["independent"], 1.0 - confidence)
        explanation = self._build_explanation(
            label,
            confidence,
            distance_delta,
            cosine,
            opposite_cosine,
            action_a,
            action_b,
            stable_a,
            stable_b,
        )

        return RelationFrameRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            relation_label=label,
            relation_confidence=float(confidence),
            hand_ids=[str(first["hand_id"]), str(second["hand_id"])],
            debug_explanation=explanation,
        )

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0))

    def _build_explanation(
        self,
        label: str,
        confidence: float,
        distance_delta: float,
        cosine: float,
        opposite_cosine: float,
        action_a: str,
        action_b: str,
        stable_a: bool,
        stable_b: bool,
    ) -> str:
        if label == self.config.relations.idle_label:
            return "independent selected because no relation score exceeded the configured confidence threshold"
        if label == "approaching":
            return f"approaching detected because inter-hand distance decreased by {distance_delta:.3f}"
        if label == "separating":
            return f"separating detected because inter-hand distance increased by {distance_delta:.3f}"
        if label == "coordinated":
            return f"coordinated detected because hand motion vectors were aligned with cosine similarity {cosine:.2f}"
        if label == "mirrored_motion":
            return f"mirrored_motion detected because opposing vectors aligned with similarity {opposite_cosine:.2f}"
        if label == "supporting":
            return (
                "supporting detected because one hand remained stable while the other executed an active action "
                f"({action_a} / {action_b})"
            )
        if label == "dual_hold":
            return "dual_hold detected because both hands were closed and stable at the same time"
        if label == "handoff_like":
            return f"handoff_like detected because release/grab actions co-occurred ({action_a} / {action_b}) nearby"
        return f"{label} detected with relation confidence {confidence:.2f}; stable states {stable_a}/{stable_b}"
