from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

from annotations import load_annotation_records, save_annotation_template
from batch_runner import build_batch_summary, discover_videos, resolve_annotation_path
from config import DEFAULT_CONFIG, AppConfig
from debug_tools import (
    DebugEvent,
    action_records_to_debug_events,
    debug_events_to_dataframe,
    dict_events_to_debug_events,
    relation_records_to_debug_events,
)
from evaluation import Evaluator
from features import FeatureExtractor
from gestures import LowLevelActionAnalyzer, LowLevelActionDecision
from grammar import ActionGrammarAnalyzer
from io_utils import prepare_output_paths, save_dataclass_records, save_frame_records
from presets import apply_preset, export_threshold_report, list_presets
from primitives import PrimitiveStateAnalyzer, PrimitiveStateRecord
from relations import InterHandRelationAnalyzer
from reporting import ReportBuilder
from segments import ActionSegment, RelationSegment, SegmentConsolidator
from tracker import HandTracker, TrackedHand
from visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layered hand action analysis pipeline for single videos or batches.")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--input", help="Single input video path.")
    source_group.add_argument("--input_dir", help="Directory containing input videos.")
    parser.add_argument("--output_dir", required=False, default="results", help="Output directory or batch root directory.")
    parser.add_argument("--annotations", help="Optional annotation file or annotation directory.")
    parser.add_argument("--preset", default="default", choices=list_presets(), help="Scenario/camera preset.")
    parser.add_argument(
        "--render_mode",
        choices=("both", "analytic", "ghost", "none"),
        default="both",
        help="Select rendered output videos.",
    )
    parser.add_argument("--debug_exports", action="store_true", help="Export debug CSV/JSON explanation files.")
    parser.add_argument("--snapshot_exports", action="store_true", help="Export representative snapshot frames.")
    parser.add_argument("--no_reports", action="store_true", help="Disable markdown/text/timeline/event reports.")
    parser.add_argument("--no_evaluation", action="store_true", help="Disable evaluation and descriptive summary outputs.")
    parser.add_argument("--disable_plots", action="store_true", help="Skip matplotlib debug plots.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap for debugging.")
    parser.add_argument("--push_axis", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Override push-axis vector.")
    parser.add_argument(
        "--frame_left_maps_to",
        choices=("Left", "Right"),
        help="Handedness prior used when both visible hands collapse to the same label.",
    )
    parser.add_argument("--swap_handedness_labels", action="store_true", help="Swap MediaPipe Left/Right labels.")
    parser.add_argument("--list_presets", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--write_annotation_template", help="Write an empty annotation template CSV and exit.")
    return parser.parse_args()


def build_feature_record(
    hand: TrackedHand,
    features: Dict[str, float],
    primitive: PrimitiveStateRecord,
    action: LowLevelActionDecision,
    frame_index: int,
    timestamp: float,
) -> Dict[str, float | str]:
    record: Dict[str, float | str] = {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "hand_id": hand.track_id,
        "handedness": hand.handedness,
        "raw_handedness": hand.raw_handedness,
        "handedness_score": hand.handedness_score,
        "tracker_confidence": hand.confidence,
        "tracker_stability": hand.stability,
        "action_label": action.action_label,
        "action_confidence": action.action_confidence,
        "action_debug_explanation": action.debug_explanation,
        "primitive_labels": "|".join(primitive.primitive_labels),
        "primitive_posture_state": primitive.posture_state,
        "primitive_transition_state": primitive.transition_state,
        "primitive_motion_state": primitive.motion_state,
        "primitive_rotation_state": primitive.rotation_state,
        "primitive_direction_state": primitive.direction_state,
    }
    for lm_idx, (x_val, y_val, z_val) in enumerate(hand.landmarks_norm):
        record[f"lm_{lm_idx}_x"] = float(x_val)
        record[f"lm_{lm_idx}_y"] = float(y_val)
        record[f"lm_{lm_idx}_z"] = float(z_val)
    for key, value in features.items():
        if key in {"frame_idx", "timestamp"}:
            continue
        record[key] = value
    return record


def configure_from_args(args: argparse.Namespace) -> AppConfig:
    config = apply_preset(deepcopy(DEFAULT_CONFIG), args.preset)
    if args.push_axis is not None:
        config.gestures.push_axis = tuple(args.push_axis)
    if args.frame_left_maps_to is not None:
        config.tracker.frame_left_maps_to = args.frame_left_maps_to
    if args.swap_handedness_labels:
        config.tracker.swap_handedness_labels = True
    if args.disable_plots:
        config.visualization.generate_plots = False
    config.debug.enable_debug_exports = bool(args.debug_exports)
    config.debug.export_snapshots = bool(args.snapshot_exports)
    config.runtime.process_analytic_video = args.render_mode in {"both", "analytic"}
    config.runtime.process_ghost_video = args.render_mode in {"both", "ghost"}
    config.runtime.generate_reports = not args.no_reports
    config.runtime.generate_evaluation = not args.no_evaluation
    return config


def save_empty_json(path: Path) -> None:
    path.write_text("[]", encoding="utf-8")


def save_json_object(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_or_empty_dataclasses(items: List[object], csv_path: Path, json_path: Path | None, columns: List[str]) -> pd.DataFrame:
    if items:
        return save_dataclass_records(items, csv_path, json_path)
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, index=False)
    if json_path is not None:
        save_empty_json(json_path)
    return df


def collect_debug_events(
    action_frame_records: List[Dict[str, object]],
    relation_frame_records: List[Dict[str, object]],
    segment_debug_events: List[Dict[str, object]],
    grammar_debug_events: List[Dict[str, object]],
    tracking_summary: Dict[str, int],
) -> List[DebugEvent]:
    events: List[DebugEvent] = []
    events.extend(action_records_to_debug_events(action_frame_records))
    events.extend(relation_records_to_debug_events(relation_frame_records))
    events.extend(dict_events_to_debug_events(segment_debug_events))
    events.extend(dict_events_to_debug_events(grammar_debug_events))
    for key, value in tracking_summary.items():
        if value:
            events.append(
                DebugEvent(
                    event_type="tracking_summary",
                    scope="tracker",
                    label=key,
                    start_frame=-1,
                    end_frame=-1,
                    reason=f"{key}={value}",
                )
            )
    return events


def process_video(
    video_path: Path,
    output_root: Path,
    config: AppConfig,
    annotation_path: Path | None = None,
    max_frames: int | None = None,
    use_video_subdir: bool = True,
) -> Dict[str, object]:
    video_output_dir = (output_root / video_path.stem) if use_video_subdir else output_root
    paths = prepare_output_paths(video_output_dir, config)
    save_json_object(paths.threshold_report_json, export_threshold_report(config))
    run_started_at = datetime.now(timezone.utc).isoformat()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = config.video.default_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video dimensions could not be determined for {video_path}")

    tracker = HandTracker(config, width, height)
    extractor = FeatureExtractor(config)
    primitive_analyzer = PrimitiveStateAnalyzer(config)
    action_analyzer = LowLevelActionAnalyzer(config)
    relation_analyzer = InterHandRelationAnalyzer(config)

    frame_records: List[Dict[str, float | str]] = []
    primitive_records: List[Dict[str, object]] = []
    action_frame_records: List[Dict[str, object]] = []
    relation_frame_records: List[Dict[str, object]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            timestamp = frame_index / fps
            hands = tracker.process_frame(frame, frame_index)
            frame_hand_records: List[Dict[str, object]] = []
            for hand in hands:
                features = extractor.compute(hand, frame_index, timestamp)
                primitive = primitive_analyzer.update(hand, features, frame_index, timestamp)
                action = action_analyzer.update(hand, primitive, features, frame_index, timestamp)
                frame_records.append(build_feature_record(hand, features, primitive, action, frame_index, timestamp))
                primitive_records.append(primitive.to_record())
                action_frame_records.append(action.to_record())
                frame_hand_records.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "hand_id": hand.track_id,
                        "handedness": hand.handedness,
                        "palm_center_x": float(hand.palm_center[0]),
                        "palm_center_y": float(hand.palm_center[1]),
                        "hand_scale": float(hand.scale),
                        "motion_dx": float(features["motion_dx"]),
                        "motion_dy": float(features["motion_dy"]),
                        "engaged_score": float(features["engaged_score"]),
                        "action_label": action.action_label,
                        "primitive_motion_state": primitive.motion_state,
                        "primitive_posture_state": primitive.posture_state,
                    }
                )
            relation_frame_records.append(
                relation_analyzer.update(frame_index, timestamp, frame_hand_records).to_record()
            )
            frame_index += 1
    finally:
        capture.release()
        tracker.close()

    frame_df = save_frame_records(frame_records, paths.frame_csv)
    primitive_df = save_frame_records(primitive_records, paths.primitive_csv)
    relation_frame_df = save_frame_records(relation_frame_records, paths.relation_frames_csv)

    consolidator = SegmentConsolidator(config)
    action_segments: List[ActionSegment] = consolidator.consolidate_actions(action_frame_records)
    relation_segments: List[RelationSegment] = consolidator.consolidate_relations(relation_frame_records)
    grammar = ActionGrammarAnalyzer(config)
    phrases = grammar.analyze(action_segments)
    tracking_summary = tracker.get_tracking_summary()

    action_segments_df = save_or_empty_dataclasses(
        action_segments,
        paths.action_segments_csv,
        paths.action_segments_json,
        ["hand_id", "handedness", "action_label", "label", "start_frame", "end_frame", "start_time", "end_time", "confidence", "mean_confidence", "notes"],
    )
    phrases_df = save_or_empty_dataclasses(
        phrases,
        paths.phrases_csv,
        paths.phrases_json,
        ["phrase_scope", "phrase_label", "component_actions", "start_frame", "end_frame", "start_time", "end_time", "confidence", "notes"],
    )
    relation_segments_df = save_or_empty_dataclasses(
        relation_segments,
        paths.relations_csv,
        paths.relations_json,
        ["relation_label", "start_frame", "end_frame", "start_time", "end_time", "confidence", "mean_confidence", "notes"],
    )

    debug_events = collect_debug_events(
        action_frame_records,
        relation_frame_records,
        consolidator.debug_events,
        grammar.debug_events,
        tracking_summary,
    )
    debug_events_df = debug_events_to_dataframe(debug_events)
    if config.debug.enable_debug_exports:
        debug_events_df.to_csv(paths.debug_events_csv, index=False)
        save_empty_json(paths.debug_events_json) if debug_events_df.empty else save_json_object(
            paths.debug_events_json,
            {"debug_events": debug_events_df.to_dict(orient="records")},
        )

    annotations = load_annotation_records(annotation_path) if annotation_path is not None else None
    evaluator = Evaluator(config)
    evaluation_result = evaluator.evaluate(
        frame_df,
        action_segments_df,
        phrases_df,
        relation_segments_df,
        annotations=annotations,
        tracking_summary=tracking_summary,
    )
    if config.runtime.generate_evaluation:
        save_json_object(paths.evaluation_summary_json, evaluation_result.summary)
        evaluation_result.metrics_table.to_csv(paths.evaluation_metrics_csv, index=False)

    reporter = ReportBuilder(config)
    event_summary_df = reporter.build_event_summary(action_segments_df, phrases_df, relation_segments_df)
    timeline_summary_df = reporter.build_timeline_summary(frame_df, relation_frame_df, phrases_df)
    if config.runtime.generate_reports:
        event_summary_df.to_csv(paths.event_summary_csv, index=False)
        timeline_summary_df.to_csv(paths.timeline_summary_csv, index=False)
        reporter.write_markdown_report(paths.markdown_report, video_path.name, evaluation_result.summary, event_summary_df, debug_events_df)
        reporter.write_text_report(paths.text_report, video_path.name, evaluation_result.summary, event_summary_df)
        reporter.export_snapshots(str(video_path), paths.snapshots_dir, event_summary_df, phrases_df, relation_segments_df, fps)

    visualizer = Visualizer(config)
    if not frame_df.empty:
        if config.runtime.process_analytic_video and config.runtime.process_ghost_video:
            visualizer.render_videos(
                str(video_path),
                frame_df,
                phrases_df,
                relation_frame_df,
                paths.analytic_video,
                paths.ghost_video,
                fps,
                (width, height),
            )
        elif config.runtime.process_analytic_video:
            visualizer.render_video_mode(
                str(video_path),
                frame_df,
                phrases_df,
                relation_frame_df,
                paths.analytic_video,
                fps,
                (width, height),
                "analytic_overlay",
            )
        elif config.runtime.process_ghost_video:
            visualizer.render_video_mode(
                str(video_path),
                frame_df,
                phrases_df,
                relation_frame_df,
                paths.ghost_video,
                fps,
                (width, height),
                "ghost_diagram",
            )

    if config.visualization.generate_plots:
        visualizer.create_debug_plots(frame_df, primitive_df, phrases_df, relation_frame_df, paths.plots_dir)

    duration = (frame_index / fps) if frame_index else 0.0
    run_metadata = {
        "input_video_name": video_path.name,
        "input_video_path": str(video_path.resolve()),
        "results_dir": str(video_output_dir.resolve()),
        "profile_name": config.runtime.profile_name,
        "processed_at_utc": run_started_at,
        "duration_seconds": duration,
        "processed_frames": frame_index,
        "fps": fps,
        "frame_size": {"width": width, "height": height},
        "render_mode": {
            "analytic": config.runtime.process_analytic_video,
            "ghost": config.runtime.process_ghost_video,
        },
        "warnings": evaluation_result.summary.get("warning_flags", []),
        "output_paths": {
            "frame_csv": str(paths.frame_csv),
            "primitive_csv": str(paths.primitive_csv),
            "action_segments_csv": str(paths.action_segments_csv),
            "phrases_csv": str(paths.phrases_csv),
            "relations_csv": str(paths.relations_csv),
            "relation_frames_csv": str(paths.relation_frames_csv),
            "analytic_video": str(paths.analytic_video),
            "ghost_video": str(paths.ghost_video),
        },
    }
    save_json_object(paths.run_metadata_json, run_metadata)
    summary = {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "duration_seconds": duration,
        "processed_frames": frame_index,
        "frame_hint": total_frame_hint,
        "hands_detected": len(frame_df["hand_id"].unique()) if not frame_df.empty else 0,
        "action_counts": json.dumps(evaluation_result.summary.get("action_counts", {}), sort_keys=True),
        "phrase_counts": json.dumps(evaluation_result.summary.get("phrase_counts", {}), sort_keys=True),
        "relation_counts": json.dumps(evaluation_result.summary.get("relation_counts", {}), sort_keys=True),
        "avg_action_duration_seconds": evaluation_result.summary.get("avg_action_duration_seconds", 0.0),
        "primitive_uncertain_frequency": evaluation_result.summary.get("primitive_uncertain_frequency", 0.0),
        "warning_flags": "|".join(evaluation_result.summary.get("warning_flags", [])),
        "profile_name": config.runtime.profile_name,
        "annotation_file": str(annotation_path) if annotation_path else "",
        "resolved_label_conflicts": tracking_summary.get("resolved_label_conflicts", 0),
        "suspected_identity_switches": tracking_summary.get("suspected_identity_switches", 0),
        "dropout_recoveries": tracking_summary.get("recovered_after_dropout", 0),
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.list_presets:
        for preset in list_presets():
            print(preset)
        return
    if args.write_annotation_template:
        save_annotation_template(args.write_annotation_template)
        print(f"Annotation template written to {args.write_annotation_template}")
        return
    if not args.input and not args.input_dir:
        raise SystemExit("Provide either --input or --input_dir.")

    config = configure_from_args(args)
    output_root = Path(args.output_dir)
    videos = discover_videos(args.input or args.input_dir, config)
    if not videos:
        raise SystemExit("No supported videos found.")

    video_summaries: List[Dict[str, object]] = []
    use_video_subdir = args.input_dir is not None
    for video_path in videos:
        annotation_path = resolve_annotation_path(args.annotations, video_path)
        summary = process_video(
            video_path=video_path,
            output_root=output_root,
            config=config,
            annotation_path=annotation_path,
            max_frames=args.max_frames,
            use_video_subdir=use_video_subdir,
        )
        video_summaries.append(summary)
        target_dir = (output_root / video_path.stem) if use_video_subdir else output_root
        print(f"Processed {video_path.name} -> {target_dir}")

    batch_summary_df = build_batch_summary(video_summaries)
    batch_summary_path = output_root / config.reporting.batch_summary_filename
    batch_summary_df.to_csv(batch_summary_path, index=False)
    print(f"Batch summary: {batch_summary_path}")


if __name__ == "__main__":
    main()
