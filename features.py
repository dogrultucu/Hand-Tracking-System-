from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict

import numpy as np

from config import AppConfig
from tracker import TrackedHand


FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


class FeatureExtractor:
    """Computes temporally-aware hand features for gesture analysis."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.history: Dict[str, Deque[Dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.config.features.temporal_window)
        )

    def compute(self, hand: TrackedHand, frame_idx: int, timestamp: float) -> Dict[str, float]:
        landmarks_px = hand.landmarks_px
        landmarks_norm = hand.landmarks_norm
        scale = max(float(hand.scale), 1e-6)
        palm_center = hand.palm_center
        wrist = landmarks_px[0]
        bbox = hand.bbox

        finger_curls = {
            name: self._finger_curl(landmarks_norm, chain)
            for name, chain in FINGER_CHAINS.items()
        }
        avg_curl = float(np.mean(list(finger_curls.values())))

        fingertip_indices = [4, 8, 12, 16, 20]
        tip_distances = {
            f"tip_distance_{tip_idx}": float(np.linalg.norm(landmarks_px[tip_idx, :2] - palm_center[:2]) / scale)
            for tip_idx in fingertip_indices
        }
        thumb_index_distance = float(np.linalg.norm(landmarks_px[4, :2] - landmarks_px[8, :2]) / scale)
        thumb_middle_distance = float(np.linalg.norm(landmarks_px[4, :2] - landmarks_px[12, :2]) / scale)

        average_tip_distance = float(np.mean(list(tip_distances.values())))
        openness = float(
            np.clip(
                (self.config.features.openness_tip_weight * (average_tip_distance / 1.65))
                + (self.config.features.openness_curl_weight * (1.0 - avg_curl)),
                0.0,
                1.0,
            )
        )
        engaged_score = float(
            np.clip(
                (0.55 * avg_curl) + (0.25 * (1.0 - thumb_index_distance)) + (0.20 * (1.0 - openness)),
                0.0,
                1.0,
            )
        )

        index_pinky_spread = float(np.linalg.norm(landmarks_px[5, :2] - landmarks_px[17, :2]) / scale)
        palm_height = float(np.linalg.norm(landmarks_px[0, :2] - landmarks_px[9, :2]) / scale)
        bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
        normalized_area = float(bbox_area / max(scale * scale * 10.0, 1.0))
        side_view_score = float(
            np.clip(
                1.0
                - (
                    0.55 * np.clip(index_pinky_spread / self.config.features.side_view_spread_threshold, 0.0, 1.0)
                    + 0.45 * np.clip(normalized_area / self.config.features.side_view_area_threshold, 0.0, 1.0)
                ),
                0.0,
                1.0,
            )
        )

        palm_axis = landmarks_norm[17] - landmarks_norm[5]
        forearm_axis = landmarks_norm[9] - landmarks_norm[0]
        palm_normal = np.cross(palm_axis, forearm_axis)
        palm_orientation = float(math.atan2(palm_axis[1], palm_axis[0]))
        palm_depth = float(np.mean(landmarks_norm[[0, 5, 9, 13, 17], 2]))

        history = self.history[hand.track_id]
        previous = history[-1] if history else None
        motion_dx = 0.0
        motion_dy = 0.0
        motion_dz = 0.0
        motion_magnitude = 0.0
        acceleration = 0.0
        curvature = 0.0
        rotation_velocity = 0.0
        posture_change = 0.0
        landmark_stability = float(np.clip(0.5 * hand.stability + 0.5 * hand.confidence, 0.0, 1.0))

        if previous is not None:
            dt = max(timestamp - previous["timestamp"], 1e-6)
            motion_dx = float((palm_center[0] - previous["palm_center_x"]) / scale)
            motion_dy = float((palm_center[1] - previous["palm_center_y"]) / scale)
            motion_dz = float((palm_depth - previous["palm_depth"]) / 0.1)
            motion_magnitude = float(math.sqrt((motion_dx * motion_dx) + (motion_dy * motion_dy) + (motion_dz * motion_dz)) / dt)
            acceleration = float((motion_magnitude - previous["motion_magnitude"]) / dt)
            rotation_velocity = float(self._wrapped_angle_delta(palm_orientation, previous["palm_orientation"]) / dt)
            posture_change = float(
                abs(avg_curl - previous["avg_finger_curl"])
                + abs(openness - previous["openness_score"])
                + abs(thumb_index_distance - previous["thumb_index_distance"])
            )

        if len(history) >= 2:
            prev_vec = np.array([history[-1]["motion_dx"], history[-1]["motion_dy"]], dtype=float)
            curr_vec = np.array([motion_dx, motion_dy], dtype=float)
            curvature = self._vector_turn_angle(prev_vec, curr_vec)

        features = {
            "frame_idx": float(frame_idx),
            "timestamp": float(timestamp),
            "wrist_x": float(wrist[0]),
            "wrist_y": float(wrist[1]),
            "palm_center_x": float(palm_center[0]),
            "palm_center_y": float(palm_center[1]),
            "bbox_x_min": float(bbox[0]),
            "bbox_y_min": float(bbox[1]),
            "bbox_x_max": float(bbox[2]),
            "bbox_y_max": float(bbox[3]),
            "hand_scale": float(scale),
            "palm_width": float(index_pinky_spread),
            "palm_height": float(palm_height),
            "thumb_index_distance": float(thumb_index_distance),
            "thumb_middle_distance": float(thumb_middle_distance),
            "thumb_curl": float(finger_curls["thumb"]),
            "index_curl": float(finger_curls["index"]),
            "middle_curl": float(finger_curls["middle"]),
            "ring_curl": float(finger_curls["ring"]),
            "pinky_curl": float(finger_curls["pinky"]),
            "avg_finger_curl": float(avg_curl),
            "openness_score": float(openness),
            "engaged_score": float(engaged_score),
            "side_view_score": float(side_view_score),
            "motion_dx": float(motion_dx),
            "motion_dy": float(motion_dy),
            "motion_dz": float(motion_dz),
            "motion_magnitude": float(motion_magnitude),
            "acceleration": float(acceleration),
            "trajectory_curvature": float(curvature),
            "palm_orientation": float(palm_orientation),
            "rotation_velocity": float(rotation_velocity),
            "palm_depth": float(palm_depth),
            "posture_change": float(posture_change),
            "hand_presence_confidence": float(hand.confidence),
            "landmark_stability_score": float(landmark_stability),
            "visibility_score": float(np.clip(0.55 * hand.confidence + 0.45 * hand.stability, 0.0, 1.0)),
            "palm_normal_x": float(palm_normal[0]),
            "palm_normal_y": float(palm_normal[1]),
            "palm_normal_z": float(palm_normal[2]),
        }
        features.update(tip_distances)

        if history and self.config.features.motion_smoothing_alpha > 0.0:
            alpha = self.config.features.motion_smoothing_alpha
            for key in ("motion_dx", "motion_dy", "motion_dz", "motion_magnitude", "rotation_velocity", "posture_change"):
                features[key] = (alpha * features[key]) + ((1.0 - alpha) * history[-1][key])

        history.append({**features, "timestamp": float(timestamp)})
        return features

    @staticmethod
    def _joint_angle(points: np.ndarray, a: int, b: int, c: int) -> float:
        vec1 = points[a] - points[b]
        vec2 = points[c] - points[b]
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-6
        cosine = float(np.clip(np.dot(vec1, vec2) / denom, -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def _finger_curl(self, landmarks_norm: np.ndarray, chain: tuple[int, int, int, int]) -> float:
        mcp, pip, dip, tip = chain
        angle_1 = self._joint_angle(landmarks_norm, mcp, pip, dip)
        angle_2 = self._joint_angle(landmarks_norm, pip, dip, tip)
        extension = np.clip(((angle_1 + angle_2) / 2.0 - 55.0) / 125.0, 0.0, 1.0)
        return float(1.0 - extension)

    @staticmethod
    def _wrapped_angle_delta(current: float, previous: float) -> float:
        delta = current - previous
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta

    @staticmethod
    def _vector_turn_angle(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
        return float(math.acos(cosine))
