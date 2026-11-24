from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import pandas as pd

from config import AppConfig


class ReportBuilder:
    """Creates review-friendly tables, reports, and optional snapshots."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_event_summary(
        self,
        action_segments_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_segments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if action_segments_df.empty:
            return pd.DataFrame(
                columns=[
                    "hand",
                    "action_label",
                    "start_time",
                    "end_time",
                    "duration",
                    "confidence",
                    "linked_grammar_phrase",
                    "linked_interhand_relation",
                    "notes",
                ]
            )

        rows: List[Dict[str, object]] = []
        for segment in action_segments_df.to_dict(orient="records"):
            rows.append(
                {
                    "hand": segment.get("hand_id", ""),
                    "action_label": segment.get("action_label", segment.get("label", "")),
                    "start_time": segment.get("start_time", 0.0),
                    "end_time": segment.get("end_time", 0.0),
                    "duration": float(segment.get("end_time", 0.0)) - float(segment.get("start_time", 0.0)),
                    "confidence": segment.get("confidence", segment.get("mean_confidence", 0.0)),
                    "linked_grammar_phrase": self._find_phrase_overlap(segment, phrase_df),
                    "linked_interhand_relation": self._find_relation_overlap(segment, relation_segments_df),
                    "notes": segment.get("notes", ""),
                }
            )
        return pd.DataFrame(rows)

    def build_timeline_summary(
        self,
        frame_df: pd.DataFrame,
        relation_frame_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame_df.empty:
            return pd.DataFrame(
                columns=[
                    "frame_index",
                    "timestamp",
                    "hand_id",
                    "handedness",
                    "primitive_labels",
                    "action_label",
                    "active_phrase_labels",
                    "relation_label",
                ]
            )

        phrase_timeline = self._expand_phrase_timeline(phrase_df)
        relation_lookup = {
            int(row.frame_index): str(row.relation_label)
            for row in relation_frame_df.itertuples(index=False)
        }
        rows: List[Dict[str, object]] = []
        for row in frame_df.itertuples(index=False):
            frame_index = int(row.frame_index)
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp": float(row.timestamp),
                    "hand_id": str(row.hand_id),
                    "handedness": str(row.handedness),
                    "primitive_labels": str(row.primitive_labels),
                    "action_label": str(row.action_label),
                    "active_phrase_labels": "|".join(phrase_timeline.get(frame_index, [])),
                    "relation_label": relation_lookup.get(frame_index, self.config.relations.idle_label),
                }
            )
        return pd.DataFrame(rows)

    def write_markdown_report(
        self,
        output_path: Path,
        video_name: str,
        evaluation_summary: Dict[str, object],
        event_summary_df: pd.DataFrame,
        debug_events_df: pd.DataFrame,
    ) -> None:
        lines = [
            f"# Analysis Report: {video_name}",
            "",
            "## Summary",
            "",
        ]
        for key, value in evaluation_summary.items():
            lines.append(f"- **{key}**: {value}")
        lines.extend(["", "## Event Preview", ""])
        if event_summary_df.empty:
            lines.append("No action events detected.")
        else:
            preview = event_summary_df.head(12)
            lines.append("| hand | action | start | end | conf | phrase | relation |")
            lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
            for row in preview.itertuples(index=False):
                lines.append(
                    f"| {row.hand} | {row.action_label} | {row.start_time:.2f} | {row.end_time:.2f} | "
                    f"{row.confidence:.2f} | {row.linked_grammar_phrase} | {row.linked_interhand_relation} |"
                )
        lines.extend(["", "## Debug Highlights", ""])
        if debug_events_df.empty:
            lines.append("No debug events recorded.")
        else:
            for row in debug_events_df.head(15).itertuples(index=False):
                lines.append(f"- `{row.event_type}` [{row.label}] {row.reason}")
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def write_text_report(
        self,
        output_path: Path,
        video_name: str,
        evaluation_summary: Dict[str, object],
        event_summary_df: pd.DataFrame,
    ) -> None:
        lines = [f"Analysis Report: {video_name}", ""]
        for key, value in evaluation_summary.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("Events:")
        if event_summary_df.empty:
            lines.append("  none")
        else:
            for row in event_summary_df.head(20).itertuples(index=False):
                lines.append(
                    f"  {row.hand} | {row.action_label} | {row.start_time:.2f}s-{row.end_time:.2f}s | "
                    f"conf={row.confidence:.2f} | phrase={row.linked_grammar_phrase} | relation={row.linked_interhand_relation}"
                )
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def export_snapshots(
        self,
        input_path: str,
        snapshots_dir: Path,
        event_summary_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_segments_df: pd.DataFrame,
        fps: float,
    ) -> List[Path]:
        if not self.config.debug.export_snapshots:
            return []

        snapshot_paths: List[Path] = []
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            return snapshot_paths
        try:
            counters: Counter[str] = Counter()
            for row in event_summary_df.itertuples(index=False):
                label = str(row.action_label)
                if label not in self.config.segments.snapshot_key_actions:
                    continue
                if counters[label] >= self.config.reporting.snapshot_max_per_category:
                    continue
                midpoint = (float(row.start_time) + float(row.end_time)) * 0.5
                frame_index = int(midpoint * fps)
                image = self._read_frame(capture, frame_index)
                if image is None:
                    continue
                path = snapshots_dir / f"{label.replace(' ', '_').replace('+', 'plus')}_{frame_index}.{self.config.reporting.snapshot_image_format}"
                cv2.imwrite(str(path), image)
                snapshot_paths.append(path)
                counters[label] += 1

            for row in phrase_df.itertuples(index=False):
                label = str(row.phrase_label)
                if counters[label] >= self.config.reporting.snapshot_max_per_category:
                    continue
                midpoint = (float(row.start_time) + float(row.end_time)) * 0.5
                frame_index = int(midpoint * fps)
                image = self._read_frame(capture, frame_index)
                if image is None:
                    continue
                path = snapshots_dir / f"phrase_{label}_{frame_index}.{self.config.reporting.snapshot_image_format}"
                cv2.imwrite(str(path), image)
                snapshot_paths.append(path)
                counters[label] += 1

            for row in relation_segments_df.itertuples(index=False):
                label = str(row.relation_label)
                if label in {self.config.relations.idle_label, "independent"}:
                    continue
                if counters[label] >= self.config.reporting.snapshot_max_per_category:
                    continue
                midpoint = (float(row.start_time) + float(row.end_time)) * 0.5
                frame_index = int(midpoint * fps)
                image = self._read_frame(capture, frame_index)
                if image is None:
                    continue
                path = snapshots_dir / f"relation_{label}_{frame_index}.{self.config.reporting.snapshot_image_format}"
                cv2.imwrite(str(path), image)
                snapshot_paths.append(path)
                counters[label] += 1
        finally:
            capture.release()
        return snapshot_paths

    def _find_phrase_overlap(self, segment: Dict[str, object], phrase_df: pd.DataFrame) -> str:
        if phrase_df.empty:
            return ""
        hits = phrase_df[
            (phrase_df["start_time"] <= float(segment.get("end_time", 0.0)))
            & (phrase_df["end_time"] >= float(segment.get("start_time", 0.0)))
            & (phrase_df["phrase_scope"] == str(segment.get("hand_id", "")))
        ]
        return "|".join(hits["phrase_label"].astype(str).tolist())

    def _find_relation_overlap(self, segment: Dict[str, object], relation_df: pd.DataFrame) -> str:
        if relation_df.empty:
            return ""
        hits = relation_df[
            (relation_df["start_time"] <= float(segment.get("end_time", 0.0)))
            & (relation_df["end_time"] >= float(segment.get("start_time", 0.0)))
        ]
        return "|".join(hits["relation_label"].astype(str).tolist())

    def _expand_phrase_timeline(self, phrase_df: pd.DataFrame) -> Dict[int, List[str]]:
        timeline: Dict[int, List[str]] = {}
        if phrase_df.empty:
            return timeline
        for row in phrase_df.itertuples(index=False):
            for frame_index in range(int(row.start_frame), int(row.end_frame) + 1):
                timeline.setdefault(frame_index, []).append(str(row.phrase_label))
        return timeline

    @staticmethod
    def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> cv2.Mat | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        return frame if ok else None
