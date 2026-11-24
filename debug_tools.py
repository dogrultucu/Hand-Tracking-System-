from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass
class DebugEvent:
    event_type: str
    scope: str
    label: str
    start_frame: int
    end_frame: int
    reason: str

    def to_record(self) -> Dict[str, object]:
        return {
            "event_type": self.event_type,
            "scope": self.scope,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "reason": self.reason,
        }


def action_records_to_debug_events(action_frame_records: Sequence[Dict[str, object]]) -> List[DebugEvent]:
    events: List[DebugEvent] = []
    for record in action_frame_records:
        explanation = str(record.get("debug_explanation", ""))
        if not explanation:
            continue
        events.append(
            DebugEvent(
                event_type="action_decision",
                scope=str(record.get("hand_id", "")),
                label=str(record.get("action_label", "")),
                start_frame=int(record.get("frame_index", -1)),
                end_frame=int(record.get("frame_index", -1)),
                reason=explanation,
            )
        )
    return events


def relation_records_to_debug_events(relation_frame_records: Sequence[Dict[str, object]]) -> List[DebugEvent]:
    events: List[DebugEvent] = []
    for record in relation_frame_records:
        explanation = str(record.get("debug_explanation", ""))
        if not explanation:
            continue
        events.append(
            DebugEvent(
                event_type="relation_decision",
                scope=str(record.get("hand_ids", "inter_hand")),
                label=str(record.get("relation_label", "")),
                start_frame=int(record.get("frame_index", -1)),
                end_frame=int(record.get("frame_index", -1)),
                reason=explanation,
            )
        )
    return events


def dict_events_to_debug_events(records: Iterable[Dict[str, object]]) -> List[DebugEvent]:
    events: List[DebugEvent] = []
    for record in records:
        events.append(
            DebugEvent(
                event_type=str(record.get("event_type", "debug")),
                scope=str(record.get("scope", "")),
                label=str(record.get("label", "")),
                start_frame=int(record.get("start_frame", -1)),
                end_frame=int(record.get("end_frame", -1)),
                reason=str(record.get("reason", "")),
            )
        )
    return events


def debug_events_to_dataframe(events: Sequence[DebugEvent]) -> pd.DataFrame:
    return pd.DataFrame([event.to_record() for event in events])
