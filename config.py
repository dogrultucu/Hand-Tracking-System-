from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

LEFT_HAND_HEX = "#06065D"
RIGHT_HAND_HEX = "#ED0101"

LEFT_HAND_RGB = (6, 6, 93)
RIGHT_HAND_RGB = (237, 1, 1)

LEFT_HAND_BGR = (93, 6, 6)
RIGHT_HAND_BGR = (1, 1, 237)


@dataclass
class VideoConfig:
    default_fps: float = 30.0
    codec: str = "mp4v"
    analytic_filename: str = "annotated.mp4"
    ghost_filename: str = "ghost_diagram.mp4"
    supported_extensions: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg")


@dataclass
class TrackerConfig:
    max_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.5
    smoothing_alpha: float = 0.35
    confidence_ema_alpha: float = 0.45
    stability_ema_alpha: float = 0.4
    assignment_max_cost: float = 2.8
    dropout_tolerance_frames: int = 8
    stale_track_removal_frames: int = 30
    label_history_size: int = 12
    use_horizontal_position_prior: bool = True
    frame_left_maps_to: str = "Right"
    swap_handedness_labels: bool = False
    max_trajectory_points: int = 40
    suspected_switch_cost_threshold: float = 0.85


@dataclass
class FeatureConfig:
    temporal_window: int = 12
    openness_tip_weight: float = 0.55
    openness_curl_weight: float = 0.45
    side_view_spread_threshold: float = 1.15
    side_view_area_threshold: float = 0.65
    motion_smoothing_alpha: float = 0.45
    depth_motion_scale: float = 0.1
    posture_smoothing_alpha: float = 0.35


@dataclass
class PrimitiveConfig:
    window: int = 8
    open_threshold: float = 0.68
    closed_threshold: float = 0.38
    opening_delta_threshold: float = 0.07
    closing_delta_threshold: float = 0.07
    stable_motion_threshold: float = 0.85
    moving_motion_threshold: float = 1.15
    rotating_threshold: float = 0.7
    rotation_direction_threshold: float = 0.45
    primitive_confirm_frames: int = 2
    uncertain_visibility_threshold: float = 0.42
    side_view_direction_min_score: float = 0.55
    side_view_vertical_motion_threshold: float = 0.075
    side_view_vertical_dominance_ratio: float = 1.2


@dataclass
class GestureConfig:
    analysis_window: int = 10
    min_confidence_for_action: float = 0.5
    engaged_score_threshold: float = 0.58
    hold_motion_threshold: float = 0.9
    move_motion_threshold: float = 1.2
    turn_rate_threshold: float = 0.75
    release_delta_threshold: float = 0.16
    grab_delta_threshold: float = 0.14
    push_axis: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    idle_label: str = "idle"


@dataclass
class SegmentConfig:
    min_action_frames: int = 4
    max_merge_gap_frames: int = 3
    suppress_idle: bool = True
    relation_min_frames: int = 5
    relation_merge_gap_frames: int = 4
    phrase_min_component_confidence: float = 0.5
    snapshot_key_actions: Tuple[str, ...] = (
        "grab",
        "turn",
        "pull + turn",
        "push + turn",
        "release + push",
    )


@dataclass
class GrammarConfig:
    max_gap_frames: int = 18
    min_phrase_confidence: float = 0.55
    allow_overlap_margin_frames: int = 4


@dataclass
class RelationConfig:
    approaching_speed_threshold: float = 0.08
    separating_speed_threshold: float = 0.08
    coordinated_similarity_threshold: float = 0.8
    mirrored_similarity_threshold: float = -0.75
    support_motion_threshold: float = 0.65
    dual_hold_min_confidence: float = 0.55
    handoff_distance_delta_threshold: float = 0.15
    relation_confidence_threshold: float = 0.5
    idle_label: str = "independent"


@dataclass
class VisualizationConfig:
    generate_plots: bool = True
    analytic_fade: float = 0.82
    ghost_fade: float = 0.25
    timeline_height: int = 110
    detection_box_thickness: int = 1
    skeleton_thickness: int = 1
    palm_marker_radius: int = 4
    landmark_radius: int = 2
    fingertip_radius: int = 3
    trajectory_thickness: int = 1
    arrow_thickness: int = 1
    echo_length: int = 10
    echo_decay: float = 0.8
    font_scale: float = 0.42
    label_thickness: int = 1
    left_hand_hex: str = LEFT_HAND_HEX
    right_hand_hex: str = RIGHT_HAND_HEX
    left_hand_rgb: Tuple[int, int, int] = LEFT_HAND_RGB
    right_hand_rgb: Tuple[int, int, int] = RIGHT_HAND_RGB
    left_hand_bgr: Tuple[int, int, int] = LEFT_HAND_BGR
    right_hand_bgr: Tuple[int, int, int] = RIGHT_HAND_BGR
    neutral_bgr: Tuple[int, int, int] = (210, 210, 210)
    timeline_background_bgr: Tuple[int, int, int] = (18, 18, 18)
    primitive_text_bgr: Tuple[int, int, int] = (220, 220, 220)
    show_relation_text: bool = True
    show_phrase_text: bool = True
    show_willow_overlay: bool = False
    show_action_sequence_text: bool = True
    action_sequence_max_labels: int = 8
    willow_baseline_frames: int = 18
    willow_min_confirm_hits: int = 3
    willow_max_missed_frames: int = 20
    willow_hough_threshold: int = 80
    willow_hough_max_gap: int = 18
    willow_min_line_length_ratio: float = 0.1
    willow_vertical_dx_ratio: float = 0.45
    willow_match_x_px: int = 56
    willow_match_y_px: int = 150
    willow_spawn_distance_ratio: float = 0.18


@dataclass
class DebugConfig:
    enable_debug_exports: bool = False
    export_snapshots: bool = False
    export_reports: bool = True
    export_timeline_csv: bool = True
    export_markdown_report: bool = True
    export_text_report: bool = True
    export_debug_json: bool = True
    export_debug_csv: bool = True


@dataclass
class EvaluationConfig:
    segment_iou_threshold: float = 0.3
    phrase_iou_threshold: float = 0.25
    relation_iou_threshold: float = 0.3
    default_annotation_format: str = "csv"


@dataclass
class ReportingConfig:
    snapshot_image_format: str = "jpg"
    snapshot_max_per_category: int = 12
    batch_summary_filename: str = "batch_summary.csv"
    threshold_report_filename: str = "thresholds_used.json"
    markdown_report_filename: str = "summary_report.md"
    text_report_filename: str = "summary_report.txt"
    event_summary_filename: str = "event_summary.csv"
    timeline_summary_filename: str = "timeline_summary.csv"
    debug_events_filename: str = "debug_events.csv"
    debug_events_json_filename: str = "debug_events.json"
    evaluation_summary_filename: str = "evaluation_summary.json"
    evaluation_metrics_csv: str = "evaluation_metrics.csv"
    snapshots_dirname: str = "snapshots"


@dataclass
class RuntimeConfig:
    profile_name: str = "default"
    process_analytic_video: bool = True
    process_ghost_video: bool = True
    generate_reports: bool = True
    generate_evaluation: bool = True


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    primitives: PrimitiveConfig = field(default_factory=PrimitiveConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)
    relations: RelationConfig = field(default_factory=RelationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = AppConfig()


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
