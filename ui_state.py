from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ReviewRunData:
    run_dir: Path
    frame_df: pd.DataFrame
    primitive_df: pd.DataFrame
    action_segments_df: pd.DataFrame
    phrase_df: pd.DataFrame
    relation_segments_df: pd.DataFrame
    relation_frame_df: pd.DataFrame
    debug_events_df: pd.DataFrame
    event_summary_df: pd.DataFrame
    timeline_summary_df: pd.DataFrame
    review_notes_df: pd.DataFrame
    threshold_report: Dict[str, object]
    run_metadata: Dict[str, object]
    evaluation_summary: Dict[str, object]
    snapshots: List[Path]


def discover_review_runs(results_dir: str | Path) -> List[Path]:
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    if (root / "frame_features.csv").exists():
        return [root]
    candidates = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and ((child / "frame_features.csv").exists() or (child / "run_metadata.json").exists()):
            candidates.append(child)
    return candidates


def load_run_data(run_dir: str | Path) -> ReviewRunData:
    run_path = Path(run_dir)
    return ReviewRunData(
        run_dir=run_path,
        frame_df=_read_csv(run_path / "frame_features.csv"),
        primitive_df=_read_csv(run_path / "primitive_states.csv"),
        action_segments_df=_read_csv(run_path / "gesture_segments.csv"),
        phrase_df=_read_csv(run_path / "action_phrases.csv"),
        relation_segments_df=_read_csv(run_path / "relation_segments.csv"),
        relation_frame_df=_read_csv(run_path / "relation_frames.csv"),
        debug_events_df=_read_csv(run_path / "debug_events.csv"),
        event_summary_df=_read_csv(run_path / "event_summary.csv"),
        timeline_summary_df=_read_csv(run_path / "timeline_summary.csv"),
        review_notes_df=_read_csv(run_path / "review_notes.csv"),
        threshold_report=_read_json(run_path / "thresholds_used.json"),
        run_metadata=_read_json(run_path / "run_metadata.json"),
        evaluation_summary=_read_json(run_path / "evaluation_summary.json"),
        snapshots=_discover_snapshots(run_path / "snapshots"),
    )


def save_review_notes(run_dir: str | Path, review_notes_df: pd.DataFrame) -> None:
    run_path = Path(run_dir)
    csv_path = run_path / "review_notes.csv"
    json_path = run_path / "review_notes.json"
    review_notes_df.to_csv(csv_path, index=False)
    json_path.write_text(review_notes_df.to_json(orient="records", indent=2), encoding="utf-8")


def flatten_thresholds(report: Dict[str, object], prefix: str = "") -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, value in report.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten_thresholds(value, full_key))
        else:
            category, _, name = full_key.partition(".")
            rows.append({"category": category, "name": name or category, "value": value})
    return rows


def append_review_note(
    review_notes_df: pd.DataFrame,
    segment_type: str,
    segment_key: str,
    verdict: str,
    alternate_label: str,
    note: str,
    selected_time: float,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "segment_type": segment_type,
                "segment_key": segment_key,
                "verdict": verdict,
                "alternate_label": alternate_label,
                "note": note,
                "selected_time": selected_time,
            }
        ]
    )
    if review_notes_df.empty:
        return row
    return pd.concat([review_notes_df, row], ignore_index=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_snapshots(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([item for item in path.iterdir() if item.is_file()])
