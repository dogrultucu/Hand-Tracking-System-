from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict

from config import AppConfig


PRESET_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "default": {},
    "frontal_close": {
        "tracker": {"smoothing_alpha": 0.3, "dropout_tolerance_frames": 6},
        "primitives": {"open_threshold": 0.7, "closed_threshold": 0.36},
        "gestures": {"push_axis": (0.0, 0.0, -1.0)},
        "visualization": {"analytic_fade": 0.86, "ghost_fade": 0.2},
    },
    "side_view_single_operator": {
        "tracker": {"smoothing_alpha": 0.42, "dropout_tolerance_frames": 10, "confidence_ema_alpha": 0.35},
        "primitives": {
            "open_threshold": 0.62,
            "closed_threshold": 0.42,
            "rotating_threshold": 0.6,
            "uncertain_visibility_threshold": 0.48,
        },
        "gestures": {"turn_rate_threshold": 0.62, "hold_motion_threshold": 0.85},
        "relations": {"relation_confidence_threshold": 0.58},
    },
    "side_view_two_hand_manipulation": {
        "tracker": {"smoothing_alpha": 0.45, "dropout_tolerance_frames": 10, "assignment_max_cost": 3.1},
        "primitives": {"moving_motion_threshold": 1.05, "rotating_threshold": 0.58},
        "segments": {"relation_min_frames": 4, "max_merge_gap_frames": 4},
        "relations": {
            "relation_confidence_threshold": 0.45,
            "coordinated_similarity_threshold": 0.72,
            "support_motion_threshold": 0.78,
        },
    },
    "high_motion_noisy_video": {
        "tracker": {"smoothing_alpha": 0.52, "stability_ema_alpha": 0.5, "dropout_tolerance_frames": 12},
        "features": {"motion_smoothing_alpha": 0.6, "posture_smoothing_alpha": 0.5},
        "primitives": {"moving_motion_threshold": 1.35, "primitive_confirm_frames": 3},
        "segments": {"min_action_frames": 5, "max_merge_gap_frames": 5},
        "grammar": {"max_gap_frames": 22},
    },
    "low_light_video": {
        "tracker": {
            "min_detection_confidence": 0.45,
            "min_tracking_confidence": 0.42,
            "smoothing_alpha": 0.46,
            "dropout_tolerance_frames": 12,
        },
        "primitives": {"uncertain_visibility_threshold": 0.5, "primitive_confirm_frames": 3},
        "gestures": {"min_confidence_for_action": 0.56},
        "visualization": {"analytic_fade": 0.9, "ghost_fade": 0.18},
    },
}


def list_presets() -> list[str]:
    return sorted(PRESET_OVERRIDES.keys())


def apply_preset(base_config: AppConfig, preset_name: str) -> AppConfig:
    config = deepcopy(base_config)
    overrides = PRESET_OVERRIDES.get(preset_name, PRESET_OVERRIDES["default"])
    for section_name, section_overrides in overrides.items():
        section = getattr(config, section_name)
        for field_name, value in section_overrides.items():
            setattr(section, field_name, value)
    config.runtime.profile_name = preset_name
    return config


def export_threshold_report(config: AppConfig) -> Dict[str, Any]:
    report = asdict(config)
    report["runtime"]["profile_name"] = config.runtime.profile_name
    return report
