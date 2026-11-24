from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas as pd

from config import AppConfig


def discover_videos(input_path: str | Path, config: AppConfig) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    videos: List[Path] = []
    for extension in config.video.supported_extensions:
        videos.extend(sorted(path.glob(f"*{extension}")))
    return sorted(set(videos))


def resolve_annotation_path(annotation_root: str | Path | None, video_path: Path) -> Path | None:
    if annotation_root is None:
        return None
    root = Path(annotation_root)
    if root.is_file():
        return root
    for extension in (".csv", ".json"):
        candidate = root / f"{video_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def build_batch_summary(video_summaries: Iterable[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for summary in video_summaries:
        rows.append(summary)
    return pd.DataFrame(rows)
