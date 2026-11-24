from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd

from config import AppConfig
from io_utils import create_video_writer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGERTIP_INDICES = {4, 8, 12, 16, 20}


@dataclass
class RenderHandState:
    hand_id: str
    handedness: str
    landmarks_px: np.ndarray
    bbox: tuple[int, int, int, int]
    palm_center: tuple[int, int]
    action_label: str
    action_confidence: float
    tracker_stability: float
    visibility_score: float
    motion_dx: float
    motion_dy: float
    primitive_posture_state: str
    primitive_motion_state: str
    primitive_rotation_state: str


@dataclass
class WillowTrack:
    willow_id: int
    handedness: str
    color: tuple[int, int, int]
    line: tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    hits: int = 1
    missed_frames: int = 0
    display_id: int | None = None

    @property
    def label(self) -> str:
        return f"W{self.display_id}" if self.display_id is not None else ""


@dataclass
class WillowOverlayState:
    baseline_lines: List[tuple[int, int, int, int]] = field(default_factory=list)
    tracks: Dict[int, WillowTrack] = field(default_factory=dict)
    next_id: int = 1
    next_display_id: int = 1


class Visualizer:
    """Pure rendering utilities for analytic and ghost-diagram video outputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.action_palette: Dict[str, tuple[int, int, int]] = {
            config.gestures.idle_label: (70, 70, 70),
            "grab": (100, 180, 255),
            "hold": (100, 220, 140),
            "turn": (255, 180, 90),
            "pull": (255, 120, 120),
            "push": (120, 190, 255),
            "pull + turn": (255, 120, 200),
            "push + turn": (180, 170, 255),
            "release + push": (160, 240, 255),
            "move + hold": (170, 255, 170),
        }

    def _ui_scale(self, frame: np.ndarray) -> float:
        return float(np.clip(frame.shape[0] / 1080.0, 1.0, 2.4))

    def _scaled_font_scale(self, frame: np.ndarray, boost: float = 1.0) -> float:
        return self.config.visualization.font_scale * self._ui_scale(frame) * boost

    def _scaled_thickness(self, frame: np.ndarray, base: int) -> int:
        return max(1, int(round(base * self._ui_scale(frame))))

    def _scaled_radius(self, frame: np.ndarray, base: int, boost: float = 1.0) -> int:
        return max(2, int(round(base * self._ui_scale(frame) * boost)))

    def _line_height(self, frame: np.ndarray, font_scale: float, thickness: int) -> int:
        (_, text_height), baseline = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return text_height + baseline + max(4, int(round(6 * self._ui_scale(frame))))

    def render_videos(
        self,
        input_path: str,
        frame_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_frame_df: pd.DataFrame,
        analytic_path: Path,
        ghost_path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self._render_video(input_path, frame_df, phrase_df, relation_frame_df, analytic_path, fps, frame_size, "analytic_overlay")
        self._render_video(input_path, frame_df, phrase_df, relation_frame_df, ghost_path, fps, frame_size, "ghost_diagram")

    def render_video_mode(
        self,
        input_path: str,
        frame_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_frame_df: pd.DataFrame,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
        mode: str,
    ) -> None:
        self._render_video(input_path, frame_df, phrase_df, relation_frame_df, output_path, fps, frame_size, mode)

    def _render_video(
        self,
        input_path: str,
        frame_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_frame_df: pd.DataFrame,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
        mode: str,
    ) -> None:
        capture = cv2.VideoCapture(input_path)
        writer = create_video_writer(output_path, fps, frame_size, self.config)
        frame_map = self._build_frame_map(frame_df, frame_size)
        phrase_timeline = self._expand_phrase_timeline(phrase_df)
        relation_timeline = {
            int(row.frame_index): (str(row.relation_label), float(row.relation_confidence))
            for row in relation_frame_df.itertuples(index=False)
        }
        action_timeline = self._build_action_timeline(frame_df)
        action_sequences = self._build_action_sequences(frame_df)
        willow_state = WillowOverlayState()
        pose_history: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.config.visualization.echo_length)
        )
        center_history: Dict[str, Deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.config.tracker.max_trajectory_points)
        )

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps if fps > 1e-6 else 0.0
                states = frame_map.get(frame_index, [])
                relation_label, relation_conf = relation_timeline.get(
                    frame_index,
                    (self.config.relations.idle_label, 0.0),
                )
                phrase_labels = phrase_timeline.get(frame_index, [])
                composed = self._compose_base(frame, mode)

                for state in states:
                    color = self._hand_color(state.handedness)
                    pose_history[state.hand_id].append(state.landmarks_px.copy())
                    center_history[state.hand_id].append(state.palm_center)
                    if mode == "ghost_diagram":
                        self._draw_pose_echoes(composed, pose_history[state.hand_id], color)
                    self._draw_hand_overlay(composed, state, color, center_history[state.hand_id], mode)

                if self.config.visualization.show_willow_overlay:
                    self._update_willow_state(frame, frame_index, states, willow_state)
                    self._draw_willow_overlay(composed, willow_state)
                self._draw_header(composed, timestamp, relation_label, relation_conf, phrase_labels, mode)
                self._draw_timeline_strip(composed, frame_index, action_timeline, action_sequences, relation_timeline, mode)
                writer.write(composed)
                frame_index += 1
        finally:
            capture.release()
            writer.release()

    def create_debug_plots(
        self,
        frame_df: pd.DataFrame,
        primitive_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_frame_df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Path]:
        output_paths: List[Path] = []
        if not frame_df.empty:
            phrase_lookup = self._expand_phrase_timeline(phrase_df)
            relation_lookup = {
                int(row.frame_index): str(row.relation_label)
                for row in relation_frame_df.itertuples(index=False)
            }
            for hand_id, hand_df in frame_df.groupby("hand_id"):
                hand_df = hand_df.sort_values("timestamp")
                primitive_hand_df = primitive_df[primitive_df["hand_id"] == hand_id].sort_values("timestamp")
                phrase_series = [
                    ",".join(phrase_lookup.get(int(frame_idx), []))
                    for frame_idx in hand_df["frame_index"].astype(int)
                ]
                relation_series = [
                    relation_lookup.get(int(frame_idx), self.config.relations.idle_label)
                    for frame_idx in hand_df["frame_index"].astype(int)
                ]
                fig, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
                axes[0].plot(hand_df["timestamp"], hand_df["openness_score"], color="#1f77b4")
                axes[0].set_ylabel("Openness")
                axes[1].plot(hand_df["timestamp"], hand_df["motion_magnitude"], color="#d62728")
                axes[1].set_ylabel("Motion")
                axes[2].plot(hand_df["timestamp"], hand_df["rotation_velocity"].abs(), color="#9467bd")
                axes[2].set_ylabel("Rotation")
                self._plot_timeline(
                    axes[3],
                    primitive_hand_df["timestamp"],
                    primitive_hand_df["primitive_labels"].fillna(""),
                    "Primitive",
                )
                self._plot_timeline(
                    axes[4],
                    hand_df["timestamp"],
                    hand_df["action_label"].fillna(self.config.gestures.idle_label),
                    "Action",
                )
                self._plot_timeline(axes[5], hand_df["timestamp"], phrase_series, "Phrase")
                self._plot_timeline(axes[6], hand_df["timestamp"], relation_series, "Relation")
                axes[6].set_xlabel("Time (s)")
                fig.suptitle(f"Analytic summary: {hand_id}")
                fig.tight_layout()
                path = output_dir / f"{hand_id}_summary.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                output_paths.append(path)

        if not relation_frame_df.empty:
            fig, ax = plt.subplots(figsize=(12, 3))
            self._plot_timeline(ax, relation_frame_df["timestamp"], relation_frame_df["relation_label"], "Relation")
            ax.set_xlabel("Time (s)")
            fig.tight_layout()
            path = output_dir / "relation_timeline.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            output_paths.append(path)

        return output_paths

    def _compose_base(self, frame: np.ndarray, mode: str) -> np.ndarray:
        if mode == "analytic_overlay":
            return cv2.convertScaleAbs(frame, alpha=self.config.visualization.analytic_fade, beta=-10)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ghost = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ghost = cv2.addWeighted(ghost, self.config.visualization.ghost_fade, np.full_like(ghost, 255), 1.0 - self.config.visualization.ghost_fade, 0)
        return ghost

    def _build_frame_map(self, frame_df: pd.DataFrame, frame_size: tuple[int, int]) -> Dict[int, List[RenderHandState]]:
        width, height = frame_size
        frame_map: Dict[int, List[RenderHandState]] = defaultdict(list)
        for row in frame_df.itertuples(index=False):
            landmarks = []
            for idx in range(21):
                x = int(getattr(row, f"lm_{idx}_x") * width)
                y = int(getattr(row, f"lm_{idx}_y") * height)
                z = float(getattr(row, f"lm_{idx}_z"))
                landmarks.append((x, y, z))
            frame_map[int(row.frame_index)].append(
                RenderHandState(
                    hand_id=str(row.hand_id),
                    handedness=str(row.handedness),
                    landmarks_px=np.asarray(landmarks, dtype=float),
                    bbox=(
                        int(row.bbox_x_min),
                        int(row.bbox_y_min),
                        int(row.bbox_x_max),
                        int(row.bbox_y_max),
                    ),
                    palm_center=(int(row.palm_center_x), int(row.palm_center_y)),
                    action_label=str(row.action_label),
                    action_confidence=float(row.action_confidence),
                    tracker_stability=float(row.tracker_stability),
                    visibility_score=float(row.visibility_score),
                    motion_dx=float(row.motion_dx),
                    motion_dy=float(row.motion_dy),
                    primitive_posture_state=str(row.primitive_posture_state),
                    primitive_motion_state=str(row.primitive_motion_state),
                    primitive_rotation_state=str(row.primitive_rotation_state),
                )
            )
        return frame_map

    def _expand_phrase_timeline(self, phrase_df: pd.DataFrame) -> Dict[int, List[str]]:
        timeline: Dict[int, List[str]] = defaultdict(list)
        if phrase_df.empty:
            return timeline
        for row in phrase_df.itertuples(index=False):
            for frame_index in range(int(row.start_frame), int(row.end_frame) + 1):
                timeline[frame_index].append(str(row.phrase_label))
        return timeline

    def _build_action_timeline(self, frame_df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
        timeline: Dict[str, Dict[int, str]] = defaultdict(dict)
        for row in frame_df.itertuples(index=False):
            timeline[str(row.handedness)][int(row.frame_index)] = str(row.action_label)
        return timeline

    def _build_action_sequences(self, frame_df: pd.DataFrame) -> Dict[str, List[tuple[int, str]]]:
        sequences: Dict[str, List[tuple[int, str]]] = defaultdict(list)
        if frame_df.empty:
            return sequences
        for handedness, hand_df in frame_df.sort_values(["handedness", "frame_index"]).groupby("handedness"):
            last_label = self.config.gestures.idle_label
            for row in hand_df.itertuples(index=False):
                label = str(row.action_label)
                if label == self.config.gestures.idle_label or label == last_label:
                    last_label = label
                    continue
                sequences[str(handedness)].append((int(row.frame_index), label))
                last_label = label
        return sequences

    def _update_willow_state(
        self,
        frame: np.ndarray,
        frame_index: int,
        states: Sequence[RenderHandState],
        willow_state: WillowOverlayState,
    ) -> None:
        detections = self._detect_willow_lines(frame)
        if frame_index < self.config.visualization.willow_baseline_frames:
            willow_state.baseline_lines = self._merge_line_candidates(willow_state.baseline_lines + detections)
            return

        candidate_lines = [
            line
            for line in detections
            if not any(self._line_matches(line, baseline) for baseline in willow_state.baseline_lines)
        ]
        self._age_willow_tracks(willow_state)
        matched_track_ids: set[int] = set()
        unmatched_lines: List[tuple[int, int, int, int]] = []

        for line in candidate_lines:
            matched_track = self._match_willow_track(line, willow_state, matched_track_ids)
            if matched_track is None:
                unmatched_lines.append(line)
                continue
            matched_track.line = line
            matched_track.last_frame = frame_index
            matched_track.missed_frames = 0
            matched_track.hits += 1
            if (
                matched_track.display_id is None
                and matched_track.hits >= self.config.visualization.willow_min_confirm_hits
            ):
                matched_track.display_id = willow_state.next_display_id
                willow_state.next_display_id += 1
            matched_track_ids.add(matched_track.willow_id)

        for line in unmatched_lines:
            owner = self._resolve_willow_owner(line, states, frame.shape[:2])
            if owner is None:
                continue
            willow_id = willow_state.next_id
            willow_state.tracks[willow_id] = WillowTrack(
                willow_id=willow_id,
                handedness=owner.handedness,
                color=self._hand_color(owner.handedness),
                line=line,
                first_frame=frame_index,
                last_frame=frame_index,
            )
            if self.config.visualization.willow_min_confirm_hits <= 1:
                willow_state.tracks[willow_id].display_id = willow_state.next_display_id
                willow_state.next_display_id += 1
            willow_state.next_id += 1

        stale_ids = [
            track_id
            for track_id, track in willow_state.tracks.items()
            if track.missed_frames > self.config.visualization.willow_max_missed_frames
        ]
        for track_id in stale_ids:
            del willow_state.tracks[track_id]

    def _detect_willow_lines(self, frame: np.ndarray) -> List[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=self.config.visualization.willow_hough_threshold,
            minLineLength=int(frame.shape[0] * self.config.visualization.willow_min_line_length_ratio),
            maxLineGap=self.config.visualization.willow_hough_max_gap,
        )
        if lines is None:
            return []

        height = frame.shape[0]
        candidates: List[tuple[int, int, int, int]] = []
        for item in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, item)
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy < int(height * self.config.visualization.willow_min_line_length_ratio):
                continue
            if dx > (dy * self.config.visualization.willow_vertical_dx_ratio):
                continue
            if y2 < int(height * 0.22):
                continue
            candidates.append((x1, y1, x2, y2))
        return self._merge_line_candidates(candidates)

    def _merge_line_candidates(self, lines: List[tuple[int, int, int, int]]) -> List[tuple[int, int, int, int]]:
        merged: List[tuple[int, int, int, int]] = []
        for line in sorted(lines, key=self._line_length, reverse=True):
            if any(self._line_matches(line, existing, x_tol=36, y_tol=120) for existing in merged):
                continue
            merged.append(line)
        return merged

    def _age_willow_tracks(self, willow_state: WillowOverlayState) -> None:
        for track in willow_state.tracks.values():
            track.missed_frames += 1

    def _match_willow_track(
        self,
        line: tuple[int, int, int, int],
        willow_state: WillowOverlayState,
        matched_track_ids: set[int],
    ) -> WillowTrack | None:
        best_track: WillowTrack | None = None
        best_cost: float | None = None
        for track_id, track in willow_state.tracks.items():
            if track_id in matched_track_ids:
                continue
            if not self._line_matches(line, track.line):
                continue
            cost = self._line_match_cost(line, track.line)
            if best_cost is None or cost < best_cost:
                best_track = track
                best_cost = cost
        return best_track

    def _resolve_willow_owner(
        self,
        line: tuple[int, int, int, int],
        states: Sequence[RenderHandState],
        frame_shape: tuple[int, int],
    ) -> RenderHandState | None:
        if not states:
            return None
        spawn_threshold = frame_shape[0] * self.config.visualization.willow_spawn_distance_ratio
        best_state: RenderHandState | None = None
        best_distance: float | None = None
        for state in states:
            palm = np.asarray(state.palm_center, dtype=float)
            line_points = np.asarray([(line[0], line[1]), (line[2], line[3])], dtype=float)
            distance = float(np.min(np.linalg.norm(line_points - palm, axis=1)))
            if best_distance is None or distance < best_distance:
                best_state = state
                best_distance = distance
        if best_state is None or best_distance is None:
            return None
        if best_distance > spawn_threshold and best_state.action_label == self.config.gestures.idle_label:
            return None
        return best_state

    def _draw_willow_overlay(self, frame: np.ndarray, willow_state: WillowOverlayState) -> None:
        overlay = frame.copy()
        font_scale = self._scaled_font_scale(frame, boost=1.05)
        thickness = self._scaled_thickness(frame, self.config.visualization.label_thickness)
        pad = max(8, int(round(12 * self._ui_scale(frame))))
        confirmed_tracks = [
            track
            for track in willow_state.tracks.values()
            if track.hits >= self.config.visualization.willow_min_confirm_hits
        ]
        ordered_tracks = sorted(confirmed_tracks, key=lambda item: (item.first_frame, item.willow_id))
        labels_by_track = {
            track.willow_id: f"W{index}"
            for index, track in enumerate(ordered_tracks, start=1)
        }
        for track in ordered_tracks:
            x1, y1, x2, y2 = track.line
            cv2.line(
                overlay,
                (x1, y1),
                (x2, y2),
                track.color,
                self._scaled_thickness(frame, 2),
                cv2.LINE_AA,
            )
            label_x = min(frame.shape[1] - pad * 3, max(x1, x2) + pad)
            label_y = int((y1 + y2) * 0.5)
            cv2.putText(
                overlay,
                labels_by_track[track.willow_id],
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                track.color,
                thickness,
                cv2.LINE_AA,
            )
        if ordered_tracks:
            summary = "willows: " + "-".join(labels_by_track[track.willow_id] for track in ordered_tracks)
            cv2.putText(
                overlay,
                summary,
                (max(14, frame.shape[1] - int(520 * self._ui_scale(frame))), max(int(72 * self._ui_scale(frame)), pad * 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.config.visualization.primitive_text_bgr,
                thickness,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

    @staticmethod
    def _line_length(line: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        return float(np.hypot(x2 - x1, y2 - y1))

    def _line_matches(
        self,
        line_a: tuple[int, int, int, int],
        line_b: tuple[int, int, int, int],
        x_tol: int | None = None,
        y_tol: int | None = None,
    ) -> bool:
        if x_tol is None:
            x_tol = self.config.visualization.willow_match_x_px
        if y_tol is None:
            y_tol = self.config.visualization.willow_match_y_px
        ax, ay = self._line_center(line_a)
        bx, by = self._line_center(line_b)
        return abs(ax - bx) <= x_tol and abs(ay - by) <= y_tol

    @staticmethod
    def _line_center(line: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = line
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _line_match_cost(
        self,
        line_a: tuple[int, int, int, int],
        line_b: tuple[int, int, int, int],
    ) -> float:
        ax, ay = self._line_center(line_a)
        bx, by = self._line_center(line_b)
        return float(abs(ax - bx) + (0.35 * abs(ay - by)) + abs(self._line_length(line_a) - self._line_length(line_b)) * 0.05)

    def _draw_pose_echoes(self, frame: np.ndarray, pose_history: Deque[np.ndarray], color: tuple[int, int, int]) -> None:
        echo_count = len(pose_history)
        for idx, landmarks in enumerate(list(pose_history)[:-1]):
            weight = self.config.visualization.echo_decay ** (echo_count - idx)
            faded_color = tuple(int(channel * weight) for channel in color)
            self._draw_skeleton(frame, landmarks.astype(int), faded_color, max(1, self.config.visualization.skeleton_thickness - 1), 0.35)

    def _draw_hand_overlay(
        self,
        frame: np.ndarray,
        state: RenderHandState,
        color: tuple[int, int, int],
        trajectory: Deque[tuple[int, int]],
        mode: str,
    ) -> None:
        alpha = float(np.clip(0.35 + (0.65 * state.visibility_score), 0.2, 1.0))
        landmarks = state.landmarks_px.astype(int)
        self._draw_skeleton(frame, landmarks, color, self.config.visualization.skeleton_thickness, alpha)
        self._draw_landmark_points(frame, landmarks, color, alpha)
        self._draw_bbox(frame, state.bbox, color, alpha)
        self._draw_palm_marker(frame, state.palm_center, color, alpha)
        self._draw_motion_arrow(frame, state, alpha)
        self._draw_trajectory(frame, trajectory, color, alpha)
        if mode == "ghost_diagram" and state.primitive_rotation_state in {"rotating", "rotating_cw", "rotating_ccw"}:
            self._draw_rotation_arc(frame, state, color, alpha)
        self._draw_label(frame, state, color, alpha, mode)

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
        alpha: float,
    ) -> None:
        overlay = frame.copy()
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_pt = tuple(landmarks[start_idx, :2])
            end_pt = tuple(landmarks[end_idx, :2])
            cv2.line(overlay, start_pt, end_pt, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _draw_landmark_points(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        overlay = frame.copy()
        for idx, (x, y, _) in enumerate(landmarks):
            base_radius = (
                self.config.visualization.fingertip_radius
                if idx in FINGERTIP_INDICES
                else self.config.visualization.landmark_radius
            )
            boost = 1.55 if idx in FINGERTIP_INDICES else 1.7
            radius = self._scaled_radius(frame, base_radius, boost)
            cv2.circle(overlay, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _draw_bbox(self, frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int], alpha: float) -> None:
        overlay = frame.copy()
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, self.config.visualization.detection_box_thickness)
        cv2.addWeighted(overlay, alpha * 0.85, frame, 1.0 - (alpha * 0.85), 0, frame)

    def _draw_palm_marker(self, frame: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], alpha: float) -> None:
        overlay = frame.copy()
        cv2.circle(
            overlay,
            center,
            self._scaled_radius(frame, self.config.visualization.palm_marker_radius, 1.2),
            color,
            self._scaled_thickness(frame, 1),
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _draw_motion_arrow(self, frame: np.ndarray, state: RenderHandState, alpha: float) -> None:
        overlay = frame.copy()
        tip = (
            int(state.palm_center[0] + (state.motion_dx * max(28, state.bbox[2] - state.bbox[0]))),
            int(state.palm_center[1] + (state.motion_dy * max(28, state.bbox[3] - state.bbox[1]))),
        )
        cv2.arrowedLine(
            overlay,
            state.palm_center,
            tip,
            (230, 230, 230),
            self.config.visualization.arrow_thickness,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.addWeighted(overlay, alpha * 0.9, frame, 1.0 - (alpha * 0.9), 0, frame)

    def _draw_rotation_arc(self, frame: np.ndarray, state: RenderHandState, color: tuple[int, int, int], alpha: float) -> None:
        overlay = frame.copy()
        clockwise = state.primitive_rotation_state == "rotating_cw"
        start_angle, end_angle = (40, 310) if clockwise else (220, 130)
        axes = (14, 14)
        cv2.ellipse(overlay, state.palm_center, axes, 0, start_angle, end_angle, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1.0 - (alpha * 0.8), 0, frame)

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: Iterable[tuple[int, int]],
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        points = list(trajectory)
        if len(points) < 2:
            return
        overlay = frame.copy()
        total = len(points)
        for idx in range(1, total):
            weight = (idx / total) * alpha
            faded = tuple(int(channel * weight) for channel in color)
            cv2.line(
                overlay,
                points[idx - 1],
                points[idx],
                faded,
                self.config.visualization.trajectory_thickness,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _draw_label(
        self,
        frame: np.ndarray,
        state: RenderHandState,
        color: tuple[int, int, int],
        alpha: float,
        mode: str,
    ) -> None:
        lines = [
            f"{state.handedness} | {state.action_label}",
            f"conf {state.action_confidence:.2f}  stab {state.tracker_stability:.2f}",
        ]
        if mode == "ghost_diagram":
            lines.append(f"{state.primitive_posture_state} / {state.primitive_motion_state}")

        font_scale = self._scaled_font_scale(frame, boost=1.1)
        thickness = self._scaled_thickness(frame, self.config.visualization.label_thickness)
        line_height = self._line_height(frame, font_scale, thickness)
        padding_x = max(8, int(round(10 * self._ui_scale(frame))))
        overlay = frame.copy()
        origin_x = max(state.bbox[0], padding_x)
        origin_y = max(state.bbox[1] - padding_x, padding_x + line_height)
        for idx, text in enumerate(lines):
            cv2.putText(
                overlay,
                text,
                (origin_x, origin_y + (idx * line_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _draw_header(
        self,
        frame: np.ndarray,
        timestamp: float,
        relation_label: str,
        relation_confidence: float,
        phrase_labels: Sequence[str],
        mode: str,
    ) -> None:
        font_scale = self._scaled_font_scale(frame, boost=1.35)
        thickness = self._scaled_thickness(frame, self.config.visualization.label_thickness)
        line_height = self._line_height(frame, font_scale, thickness)
        margin_x = max(14, int(round(18 * self._ui_scale(frame))))
        margin_y = max(14, int(round(18 * self._ui_scale(frame))))
        lines = [f"{mode}  t={timestamp:.2f}s"]
        colors = [self.config.visualization.neutral_bgr]
        if self.config.visualization.show_relation_text:
            lines.append(f"relation: {relation_label} ({relation_confidence:.2f})")
            colors.append(self.config.visualization.primitive_text_bgr)
        if self.config.visualization.show_phrase_text and phrase_labels:
            lines.append("phrase: " + ", ".join(sorted(set(phrase_labels))))
            colors.append(self.config.visualization.primitive_text_bgr)

        overlay = frame.copy()
        for idx, (text, color) in enumerate(zip(lines, colors)):
            cv2.putText(
                overlay,
                text,
                (margin_x, margin_y + ((idx + 1) * line_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    def _draw_timeline_strip(
        self,
        frame: np.ndarray,
        frame_index: int,
        action_timeline: Dict[str, Dict[int, str]],
        action_sequences: Dict[str, List[tuple[int, str]]],
        relation_timeline: Dict[int, tuple[str, float]],
        mode: str,
    ) -> None:
        ui_scale = self._ui_scale(frame)
        font_scale = self._scaled_font_scale(frame, boost=0.92)
        thickness = self._scaled_thickness(frame, self.config.visualization.label_thickness)
        line_height = self._line_height(frame, font_scale, thickness)
        bar_height = max(6, int(round(7 * ui_scale)))
        row_gap = max(8, int(round(10 * ui_scale)))
        margin = max(8, int(round(10 * ui_scale)))
        sequence_rows = 2 if self.config.visualization.show_action_sequence_text else 0
        strip_h = max(
            self.config.visualization.timeline_height,
            margin * 2 + (row_gap * 2) + (bar_height * 3) + (line_height * sequence_rows),
        )
        width = frame.shape[1]
        overlay = frame.copy()
        y0 = frame.shape[0] - strip_h
        cv2.rectangle(overlay, (0, y0), (width, frame.shape[0]), self.config.visualization.timeline_background_bgr, -1)
        window = min(120, max(frame_index + 1, 1))
        start = max(frame_index - window + 1, 0)
        rows = [("Left", 0), ("Right", 1)]
        label_x = margin
        timeline_x0 = margin + max(18, int(round(22 * ui_scale)))
        timeline_width = max(24, width - timeline_x0 - margin)
        for handedness, row_idx in rows:
            row_y = y0 + margin + (row_idx * row_gap) + ((row_idx + 1) * bar_height)
            cv2.putText(
                overlay,
                handedness[0],
                (label_x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self._hand_color(handedness),
                thickness,
                cv2.LINE_AA,
            )
            timeline = action_timeline.get(handedness, {})
            for offset, idx in enumerate(range(start, frame_index + 1)):
                x0 = timeline_x0 + int((timeline_width * offset) / max(window, 1))
                x1 = timeline_x0 + int((timeline_width * (offset + 1)) / max(window, 1))
                label = timeline.get(idx, self.config.gestures.idle_label)
                color = self.action_palette.get(label, self.config.visualization.neutral_bgr)
                cv2.rectangle(overlay, (x0, row_y - bar_height), (x1, row_y), color, -1)
        relation_y = y0 + margin + (2 * row_gap) + (3 * bar_height)
        for offset, idx in enumerate(range(start, frame_index + 1)):
            x0 = timeline_x0 + int((timeline_width * offset) / max(window, 1))
            x1 = timeline_x0 + int((timeline_width * (offset + 1)) / max(window, 1))
            label = relation_timeline.get(idx, (self.config.relations.idle_label, 0.0))[0]
            color = self.action_palette.get("move + hold", (140, 140, 140)) if label != self.config.relations.idle_label else (80, 80, 80)
            cv2.rectangle(overlay, (x0, relation_y - bar_height), (x1, relation_y), color, -1)
        cv2.putText(
            overlay,
            "R",
            (label_x, relation_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.config.visualization.primitive_text_bgr,
            thickness,
            cv2.LINE_AA,
        )
        if self.config.visualization.show_action_sequence_text:
            text_y = relation_y + line_height
            for handedness in ("Left", "Right"):
                labels = [
                    label
                    for transition_frame, label in action_sequences.get(handedness, [])
                    if transition_frame <= frame_index
                ]
                labels = labels[-self.config.visualization.action_sequence_max_labels :]
                if not labels:
                    text = f"{handedness[0]}: idle"
                else:
                    text = f"{handedness[0]}: " + " -> ".join(labels)
                (text_width, text_height), baseline = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness,
                )
                box_pad_x = max(6, int(round(8 * ui_scale)))
                box_pad_y = max(4, int(round(5 * ui_scale)))
                box_x0 = max(timeline_x0 - box_pad_x, 0)
                box_y0 = max(text_y - text_height - box_pad_y, y0)
                box_x1 = min(timeline_x0 + text_width + box_pad_x, width - margin)
                box_y1 = min(text_y + baseline + box_pad_y, frame.shape[0] - margin)
                cv2.rectangle(
                    overlay,
                    (box_x0, box_y0),
                    (box_x1, box_y1),
                    (248, 248, 248),
                    -1,
                )
                cv2.putText(
                    overlay,
                    text,
                    (timeline_x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    self._hand_color(handedness),
                    thickness,
                    cv2.LINE_AA,
                )
                text_y += line_height
        blend = 0.92 if mode == "analytic_overlay" else 0.97
        cv2.addWeighted(overlay, blend, frame, 1.0 - blend, 0, frame)

    def _plot_timeline(self, axis: plt.Axes, times: Sequence[float], labels: Sequence[str], ylabel: str) -> None:
        unique_labels = [label for label in pd.Series(labels).fillna("").astype(str).unique() if label]
        if not unique_labels:
            unique_labels = ["none"]
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        values = [label_to_index.get(str(label), -1) for label in labels]
        axis.step(times, values, where="post", color="#444444")
        axis.set_yticks(list(label_to_index.values()))
        axis.set_yticklabels(list(label_to_index.keys()))
        axis.set_ylabel(ylabel)

    def _hand_color(self, handedness: str) -> tuple[int, int, int]:
        if handedness == "Left":
            return self.config.visualization.left_hand_bgr
        if handedness == "Right":
            return self.config.visualization.right_hand_bgr
        return self.config.visualization.neutral_bgr
