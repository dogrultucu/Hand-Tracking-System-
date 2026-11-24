from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import pandas as pd

from config import AppConfig, ensure_output_dir


class OutputArtifacts:
    def __init__(self, output_dir: Path, config: AppConfig) -> None:
        self.output_dir = output_dir
        self.annotated_video = output_dir / config.video.analytic_filename
        self.analytic_video = self.annotated_video
        self.ghost_video = output_dir / config.video.ghost_filename
        self.frame_csv = output_dir / "frame_features.csv"
        self.primitive_csv = output_dir / "primitive_states.csv"
        self.action_segments_csv = output_dir / "gesture_segments.csv"
        self.action_segments_json = output_dir / "gesture_segments.json"
        self.phrases_csv = output_dir / "action_phrases.csv"
        self.phrases_json = output_dir / "action_phrases.json"
        self.relations_csv = output_dir / "relation_segments.csv"
        self.relations_json = output_dir / "relation_segments.json"
        self.relation_frames_csv = output_dir / "relation_frames.csv"
        self.event_summary_csv = output_dir / config.reporting.event_summary_filename
        self.timeline_summary_csv = output_dir / config.reporting.timeline_summary_filename
        self.debug_events_csv = output_dir / config.reporting.debug_events_filename
        self.debug_events_json = output_dir / config.reporting.debug_events_json_filename
        self.evaluation_summary_json = output_dir / config.reporting.evaluation_summary_filename
        self.evaluation_metrics_csv = output_dir / config.reporting.evaluation_metrics_csv
        self.threshold_report_json = output_dir / config.reporting.threshold_report_filename
        self.run_metadata_json = output_dir / "run_metadata.json"
        self.review_notes_csv = output_dir / "review_notes.csv"
        self.review_notes_json = output_dir / "review_notes.json"
        self.markdown_report = output_dir / config.reporting.markdown_report_filename
        self.text_report = output_dir / config.reporting.text_report_filename
        self.plots_dir = output_dir / "plots"
        self.snapshots_dir = output_dir / config.reporting.snapshots_dirname
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)


def prepare_output_paths(output_dir: str | Path, config: AppConfig) -> OutputArtifacts:
    root = ensure_output_dir(output_dir)
    return OutputArtifacts(root, config)


def create_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int], config: AppConfig) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*config.video.codec)
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def save_frame_records(frame_records: List[dict], output_path: Path) -> pd.DataFrame:
    frame_df = pd.DataFrame(frame_records)
    frame_df.to_csv(output_path, index=False)
    return frame_df


def save_json_records(records: Sequence[dict], json_path: Path) -> None:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2)


def dataclasses_to_records(items: Iterable[object]) -> List[dict]:
    records: List[dict] = []
    for item in items:
        if hasattr(item, "to_record"):
            records.append(item.to_record())  # type: ignore[call-arg]
        elif is_dataclass(item):
            records.append(asdict(item))
        else:
            raise TypeError(f"Unsupported record type: {type(item)!r}")
    return records


def save_dataclass_records(items: Iterable[object], csv_path: Path, json_path: Path | None = None) -> pd.DataFrame:
    records = dataclasses_to_records(list(items))
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    if json_path is not None:
        save_json_records(records, json_path)
    return df
