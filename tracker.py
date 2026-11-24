from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Dict, Iterable, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig


@dataclass
class Detection:
    raw_label: str
    raw_score: float
    landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    bbox: Tuple[int, int, int, int]
    palm_center: np.ndarray
    scale: float


@dataclass
class TrackedHand:
    track_id: str
    handedness: str
    raw_handedness: str
    handedness_score: float
    landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    bbox: Tuple[int, int, int, int]
    palm_center: np.ndarray
    wrist: np.ndarray
    scale: float
    confidence: float
    stability: float
    missing_frames: int
    trajectory: List[Tuple[int, int]]


@dataclass
class TrackMemory:
    numeric_id: int
    stabilized_label: str
    smoothed_landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    bbox: Tuple[int, int, int, int]
    palm_center: np.ndarray
    wrist: np.ndarray
    scale: float
    raw_label_votes: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    raw_confidence: float = 0.0
    confidence_ema: float = 0.0
    stability_ema: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    missing_frames: int = 0
    last_seen_frame: int = -1
    trajectory: deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=40))
    visible_in_frame: bool = False
    raw_label: str = "Unknown"

    @property
    def track_id(self) -> str:
        return f"hand_{self.numeric_id}"


class HandTracker:
    """Runs MediaPipe Hands and stabilizes identities across time."""

    def __init__(self, config: AppConfig, frame_width: int, frame_height: int) -> None:
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.tracker.max_hands,
            model_complexity=config.tracker.model_complexity,
            min_detection_confidence=config.tracker.min_detection_confidence,
            min_tracking_confidence=config.tracker.min_tracking_confidence,
        )
        self.tracks: Dict[int, TrackMemory] = {}
        self.next_track_id = 0
        self.stats = {
            "spawned_tracks": 0,
            "resolved_label_conflicts": 0,
            "suspected_identity_switches": 0,
            "recovered_after_dropout": 0,
        }

    def close(self) -> None:
        self.hands.close()

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> List[TrackedHand]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detections = self._extract_detections(results)
        assignments, unmatched_detections, unmatched_tracks = self._assign_detections(detections)

        for track in self.tracks.values():
            track.visible_in_frame = False

        visible_track_ids: List[int] = []
        for det_idx, track_id in assignments:
            track = self.tracks[track_id]
            self._update_track(track, detections[det_idx], frame_idx)
            visible_track_ids.append(track_id)

        for det_idx in unmatched_detections:
            track = self._spawn_track(detections[det_idx], frame_idx)
            visible_track_ids.append(track.numeric_id)

        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.missing_frames += 1
            track.visible_in_frame = False
            track.confidence_ema *= 0.97

        self._drop_stale_tracks(frame_idx)
        self._resolve_label_conflicts(visible_track_ids)
        return self._build_visible_hands(visible_track_ids)

    def _extract_detections(self, results: object) -> List[Detection]:
        detections: List[Detection] = []
        hand_landmarks = getattr(results, "multi_hand_landmarks", None)
        handedness_list = getattr(results, "multi_handedness", None)

        if not hand_landmarks or not handedness_list:
            return detections

        for idx, landmark_list in enumerate(hand_landmarks):
            classification = handedness_list[idx].classification[0]
            raw_label = classification.label
            if self.config.tracker.swap_handedness_labels:
                raw_label = "Left" if raw_label == "Right" else "Right"
            raw_score = float(classification.score)

            landmarks_px: List[List[float]] = []
            landmarks_norm: List[List[float]] = []
            for lm in landmark_list.landmark:
                x_px = float(lm.x * self.frame_width)
                y_px = float(lm.y * self.frame_height)
                landmarks_px.append([x_px, y_px, float(lm.z)])
                landmarks_norm.append([float(lm.x), float(lm.y), float(lm.z)])

            px = np.asarray(landmarks_px, dtype=float)
            norm = np.asarray(landmarks_norm, dtype=float)
            x_min = int(np.clip(np.min(px[:, 0]), 0, self.frame_width - 1))
            y_min = int(np.clip(np.min(px[:, 1]), 0, self.frame_height - 1))
            x_max = int(np.clip(np.max(px[:, 0]), 0, self.frame_width - 1))
            y_max = int(np.clip(np.max(px[:, 1]), 0, self.frame_height - 1))
            bbox = (x_min, y_min, x_max, y_max)

            palm_indices = np.array([0, 5, 9, 13, 17], dtype=int)
            palm_center = np.mean(px[palm_indices], axis=0)
            palm_width = np.linalg.norm(px[5, :2] - px[17, :2])
            middle_length = np.linalg.norm(px[0, :2] - px[9, :2])
            scale = max(palm_width, middle_length, 1.0)
            detections.append(
                Detection(
                    raw_label=raw_label,
                    raw_score=raw_score,
                    landmarks_px=px,
                    landmarks_norm=norm,
                    bbox=bbox,
                    palm_center=palm_center,
                    scale=scale,
                )
            )

        return detections

    def _assign_detections(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missing_frames <= self.config.tracker.dropout_tolerance_frames
        ]
        if not detections:
            return [], [], track_ids
        if not track_ids:
            return [], list(range(len(detections))), []

        candidates: List[Tuple[int, float, List[Tuple[int, int]]]] = []
        max_pairs = min(len(detections), len(track_ids))
        for pair_count in range(1, max_pairs + 1):
            for det_indices in combinations(range(len(detections)), pair_count):
                for track_perm in permutations(track_ids, pair_count):
                    total_cost = 0.0
                    mapping: List[Tuple[int, int]] = []
                    valid = True
                    for det_idx, track_id in zip(det_indices, track_perm):
                        cost = self._association_cost(self.tracks[track_id], detections[det_idx])
                        if cost > self.config.tracker.assignment_max_cost:
                            valid = False
                            break
                        total_cost += cost
                        mapping.append((det_idx, track_id))
                    if valid:
                        candidates.append((pair_count, total_cost, mapping))

        if not candidates:
            return [], list(range(len(detections))), track_ids

        candidates.sort(key=lambda item: (-item[0], item[1]))
        assignments = candidates[0][2]
        for det_idx, track_id in assignments:
            cost = self._association_cost(self.tracks[track_id], detections[det_idx])
            if cost > self.config.tracker.suspected_switch_cost_threshold:
                self.stats["suspected_identity_switches"] += 1
        assigned_detections = {det_idx for det_idx, _ in assignments}
        assigned_tracks = {track_id for _, track_id in assignments}
        unmatched_detections = [idx for idx in range(len(detections)) if idx not in assigned_detections]
        unmatched_tracks = [track_id for track_id in track_ids if track_id not in assigned_tracks]
        return assignments, unmatched_detections, unmatched_tracks

    def _association_cost(self, track: TrackMemory, detection: Detection) -> float:
        predicted_center = track.palm_center + (track.velocity * max(track.missing_frames, 1))
        center_cost = np.linalg.norm(predicted_center[:2] - detection.palm_center[:2]) / max(track.scale, detection.scale, 1.0)
        landmark_cost = np.mean(
            np.linalg.norm(track.smoothed_landmarks_px[:, :2] - detection.landmarks_px[:, :2], axis=1)
        ) / max(track.scale, detection.scale, 1.0)
        label_penalty = 0.0 if detection.raw_label == track.stabilized_label else 0.35
        miss_penalty = 0.12 * track.missing_frames
        return (0.6 * center_cost) + (0.3 * landmark_cost) + label_penalty + miss_penalty

    def _spawn_track(self, detection: Detection, frame_idx: int) -> TrackMemory:
        track = TrackMemory(
            numeric_id=self.next_track_id,
            stabilized_label=detection.raw_label,
            smoothed_landmarks_px=detection.landmarks_px.copy(),
            landmarks_norm=detection.landmarks_norm.copy(),
            bbox=detection.bbox,
            palm_center=detection.palm_center.copy(),
            wrist=detection.landmarks_px[0].copy(),
            scale=float(detection.scale),
            raw_confidence=float(detection.raw_score),
            confidence_ema=float(detection.raw_score),
            stability_ema=1.0,
            visible_in_frame=True,
            raw_label=detection.raw_label,
        )
        track.raw_label_votes = deque([detection.raw_label], maxlen=self.config.tracker.label_history_size)
        track.trajectory = deque(
            [(int(detection.palm_center[0]), int(detection.palm_center[1]))],
            maxlen=self.config.tracker.max_trajectory_points,
        )
        track.last_seen_frame = frame_idx
        self.tracks[track.numeric_id] = track
        self.next_track_id += 1
        self.stats["spawned_tracks"] += 1
        return track

    def _update_track(self, track: TrackMemory, detection: Detection, frame_idx: int) -> None:
        prev_palm = track.palm_center.copy()
        alpha = self.config.tracker.smoothing_alpha
        track.smoothed_landmarks_px = (
            (1.0 - alpha) * track.smoothed_landmarks_px + alpha * detection.landmarks_px
        )
        track.landmarks_norm = detection.landmarks_norm.copy()
        track.palm_center = np.mean(track.smoothed_landmarks_px[[0, 5, 9, 13, 17]], axis=0)
        track.wrist = track.smoothed_landmarks_px[0].copy()
        track.scale = float(max(detection.scale, 1.0))
        track.bbox = detection.bbox
        frame_delta = max(frame_idx - track.last_seen_frame, 1)
        track.velocity = (track.palm_center - prev_palm) / frame_delta
        if track.missing_frames > 0:
            self.stats["recovered_after_dropout"] += 1
        track.missing_frames = 0
        track.last_seen_frame = frame_idx
        track.visible_in_frame = True
        track.raw_label = detection.raw_label
        track.raw_confidence = detection.raw_score
        track.raw_label_votes.append(detection.raw_label)

        conf_alpha = self.config.tracker.confidence_ema_alpha
        stab_alpha = self.config.tracker.stability_ema_alpha
        track.confidence_ema = (1.0 - conf_alpha) * track.confidence_ema + conf_alpha * detection.raw_score
        raw_noise = np.mean(
            np.linalg.norm(track.smoothed_landmarks_px[:, :2] - detection.landmarks_px[:, :2], axis=1)
        ) / max(track.scale, 1.0)
        stability = float(np.clip(1.0 - (raw_noise / 0.45), 0.0, 1.0))
        continuity_bonus = float(np.clip(1.0 - (np.linalg.norm(track.velocity[:2]) / max(track.scale, 1.0)), 0.0, 1.0))
        track.stability_ema = (1.0 - stab_alpha) * track.stability_ema + stab_alpha * ((0.7 * stability) + (0.3 * continuity_bonus))
        track.stabilized_label = self._vote_label(track)
        track.trajectory.append((int(track.palm_center[0]), int(track.palm_center[1])))

    def _vote_label(self, track: TrackMemory) -> str:
        votes = Counter(track.raw_label_votes)
        if not votes:
            return track.stabilized_label
        top_label, top_count = votes.most_common(1)[0]
        if len(votes) == 1:
            return top_label
        current_count = votes.get(track.stabilized_label, 0)
        return track.stabilized_label if current_count >= top_count - 1 else top_label

    def _resolve_label_conflicts(self, visible_track_ids: Iterable[int]) -> None:
        visible_tracks = [self.tracks[track_id] for track_id in visible_track_ids if track_id in self.tracks]
        if len(visible_tracks) != 2 or not self.config.tracker.use_horizontal_position_prior:
            return

        labels = {track.stabilized_label for track in visible_tracks}
        if len(labels) == 2:
            return

        frame_left_label = self.config.tracker.frame_left_maps_to
        frame_right_label = "Left" if frame_left_label == "Right" else "Right"
        ordered = sorted(visible_tracks, key=lambda track: track.palm_center[0])
        ordered[0].stabilized_label = frame_left_label
        ordered[1].stabilized_label = frame_right_label
        self.stats["resolved_label_conflicts"] += 1

    def _drop_stale_tracks(self, frame_idx: int) -> None:
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if (frame_idx - track.last_seen_frame) > self.config.tracker.stale_track_removal_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def _build_visible_hands(self, visible_track_ids: Iterable[int]) -> List[TrackedHand]:
        hands: List[TrackedHand] = []
        for track_id in visible_track_ids:
            track = self.tracks.get(track_id)
            if track is None or not track.visible_in_frame:
                continue
            hands.append(
                TrackedHand(
                    track_id=track.track_id,
                    handedness=track.stabilized_label,
                    raw_handedness=track.raw_label,
                    handedness_score=float(track.confidence_ema),
                    landmarks_px=track.smoothed_landmarks_px.copy(),
                    landmarks_norm=track.landmarks_norm.copy(),
                    bbox=track.bbox,
                    palm_center=track.palm_center.copy(),
                    wrist=track.wrist.copy(),
                    scale=float(track.scale),
                    confidence=float(track.confidence_ema),
                    stability=float(track.stability_ema),
                    missing_frames=int(track.missing_frames),
                    trajectory=list(track.trajectory),
                )
            )
        hands.sort(key=lambda hand: hand.palm_center[0])
        return hands

    def get_tracking_summary(self) -> Dict[str, int]:
        return dict(self.stats)
