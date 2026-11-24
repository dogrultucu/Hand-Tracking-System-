from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


GRAMMAR_HEX = "#6B7280"
RELATION_HEX = "#A16207"


def build_timeline_dataframe(
    frame_df: pd.DataFrame,
    primitive_df: pd.DataFrame,
    action_segments_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    relation_segments_df: pd.DataFrame,
    left_hex: str,
    right_hex: str,
) -> pd.DataFrame:
    segments: List[Dict[str, object]] = []
    segments.extend(_primitive_segments(primitive_df, left_hex, right_hex))
    segments.extend(_action_segments(action_segments_df, left_hex, right_hex))
    segments.extend(_phrase_segments(phrase_df))
    segments.extend(_relation_segments(relation_segments_df))
    return pd.DataFrame(segments)


def make_timeline_figure(timeline_df: pd.DataFrame) -> go.Figure:
    if timeline_df.empty:
        fig = go.Figure()
        fig.update_layout(height=360, template="plotly_white", title="No timeline data available")
        return fig

    fig = px.timeline(
        timeline_df,
        x_start="start_time",
        x_end="end_time",
        y="track",
        color="color_name",
        color_discrete_map={
            row["color_name"]: row["color_value"]
            for row in timeline_df[["color_name", "color_value"]].drop_duplicates().to_dict(orient="records")
        },
        hover_data={
            "label": True,
            "scope": True,
            "confidence": ":.2f",
            "segment_type": True,
            "color_name": False,
            "color_value": False,
        },
        text="label",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text="Track Colors",
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle", marker_line_width=0.5)
    return fig


def _primitive_segments(primitive_df: pd.DataFrame, left_hex: str, right_hex: str) -> List[Dict[str, object]]:
    if primitive_df.empty:
        return []
    segments: List[Dict[str, object]] = []
    for hand_id, hand_df in primitive_df.groupby("hand_id"):
        ordered = hand_df.sort_values("timestamp")
        current = None
        for row in ordered.itertuples(index=False):
            label = str(row.primitive_labels)
            track = f"{row.handedness} primitives"
            if current and current["label"] == label:
                current["end_time"] = float(row.timestamp)
                current["end_frame"] = int(row.frame_index)
            else:
                if current:
                    segments.append(current)
                current = {
                    "segment_key": f"primitive::{hand_id}::{int(row.frame_index)}",
                    "segment_type": "primitive",
                    "scope": str(row.handedness),
                    "hand_id": str(hand_id),
                    "track": track,
                    "label": label or "none",
                    "start_time": float(row.timestamp),
                    "end_time": float(row.timestamp),
                    "start_frame": int(row.frame_index),
                    "end_frame": int(row.frame_index),
                    "confidence": 0.75,
                    "notes": str(row.primitive_confidences),
                    "color_name": "Left hand" if str(row.handedness) == "Left" else "Right hand",
                    "color_value": left_hex if str(row.handedness) == "Left" else right_hex,
                }
        if current:
            segments.append(current)
    return segments


def _action_segments(action_segments_df: pd.DataFrame, left_hex: str, right_hex: str) -> List[Dict[str, object]]:
    if action_segments_df.empty:
        return []
    segments = []
    for row in action_segments_df.itertuples(index=False):
        handedness = str(getattr(row, "handedness", ""))
        segments.append(
            {
                "segment_key": f"action::{row.hand_id}::{row.start_frame}",
                "segment_type": "action",
                "scope": handedness,
                "hand_id": str(row.hand_id),
                "track": f"{handedness} actions",
                "label": str(getattr(row, "action_label", getattr(row, "label", ""))),
                "start_time": float(row.start_time),
                "end_time": float(row.end_time),
                "start_frame": int(row.start_frame),
                "end_frame": int(row.end_frame),
                "confidence": float(getattr(row, "confidence", 0.0)),
                "notes": str(getattr(row, "notes", "")),
                "color_name": "Left hand" if handedness == "Left" else "Right hand",
                "color_value": left_hex if handedness == "Left" else right_hex,
            }
        )
    return segments


def _phrase_segments(phrase_df: pd.DataFrame) -> List[Dict[str, object]]:
    if phrase_df.empty:
        return []
    return [
        {
            "segment_key": f"phrase::{row.phrase_scope}::{row.start_frame}",
            "segment_type": "phrase",
            "scope": str(row.phrase_scope),
            "hand_id": str(row.phrase_scope),
            "track": "Grammar phrases",
            "label": str(row.phrase_label),
            "start_time": float(row.start_time),
            "end_time": float(row.end_time),
            "start_frame": int(row.start_frame),
            "end_frame": int(row.end_frame),
            "confidence": float(getattr(row, "confidence", 0.0)),
            "notes": str(getattr(row, "notes", "")),
            "color_name": "Grammar",
            "color_value": GRAMMAR_HEX,
        }
        for row in phrase_df.itertuples(index=False)
    ]


def _relation_segments(relation_segments_df: pd.DataFrame) -> List[Dict[str, object]]:
    if relation_segments_df.empty:
        return []
    return [
        {
            "segment_key": f"relation::inter_hand::{row.start_frame}",
            "segment_type": "relation",
            "scope": "inter_hand",
            "hand_id": "inter_hand",
            "track": "Inter-hand relations",
            "label": str(row.relation_label),
            "start_time": float(row.start_time),
            "end_time": float(row.end_time),
            "start_frame": int(row.start_frame),
            "end_frame": int(row.end_frame),
            "confidence": float(getattr(row, "confidence", 0.0)),
            "notes": str(getattr(row, "notes", "")),
            "color_name": "Relation",
            "color_value": RELATION_HEX,
        }
        for row in relation_segments_df.itertuples(index=False)
    ]
