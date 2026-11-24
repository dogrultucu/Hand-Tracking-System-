from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


ANNOTATION_COLUMNS = [
    "target_type",
    "scope",
    "label",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "confidence",
    "notes",
]


@dataclass
class AnnotationRecord:
    target_type: str
    scope: str
    label: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float = 1.0
    notes: str = ""

    def to_record(self) -> Dict[str, object]:
        return {
            "target_type": self.target_type,
            "scope": self.scope,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "notes": self.notes,
        }


def load_annotation_records(path: str | Path) -> List[AnnotationRecord]:
    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if annotation_path.suffix.lower() == ".json":
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        return [AnnotationRecord(**item) for item in data]

    df = pd.read_csv(annotation_path)
    for column in ANNOTATION_COLUMNS:
        if column not in df.columns:
            if column == "confidence":
                df[column] = 1.0
            elif column == "notes":
                df[column] = ""
            else:
                raise ValueError(f"Missing annotation column '{column}' in {annotation_path}")
    return [AnnotationRecord(**row) for row in df[ANNOTATION_COLUMNS].to_dict(orient="records")]


def save_annotation_template(path: str | Path) -> None:
    pd.DataFrame(columns=ANNOTATION_COLUMNS).to_csv(path, index=False)
