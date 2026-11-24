from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from config import AppConfig


@dataclass
class ActionSegment:
    hand_id: str
    handedness: str
    label: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    notes: str = ""

    @property
    def mean_confidence(self) -> float:
        return self.confidence

    def to_record(self) -> Dict[str, object]:
        return {
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "action_label": self.label,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "mean_confidence": self.mean_confidence,
            "notes": self.notes,
        }


@dataclass
class RelationSegment:
    relation_label: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    notes: str = ""

    @property
    def mean_confidence(self) -> float:
        return self.confidence

    def to_record(self) -> Dict[str, object]:
        return {
            "relation_label": self.relation_label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "mean_confidence": self.mean_confidence,
            "notes": self.notes,
        }


class SegmentConsolidator:
    """Builds stable segments from frame-wise low-level action and relation labels."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.debug_events: List[Dict[str, object]] = []

    def consolidate_actions(self, action_records: Sequence[Dict[str, object]]) -> List[ActionSegment]:
        self.debug_events = []
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for record in action_records:
            grouped.setdefault(str(record["hand_id"]), []).append(record)

        segments: List[ActionSegment] = []
        for hand_id, records in grouped.items():
            ordered = sorted(records, key=lambda item: int(item["frame_index"]))
            candidate_segments = self._build_action_candidates(ordered)
            candidate_segments = self._merge_action_segments(candidate_segments)
            for segment in candidate_segments:
                duration = segment.end_frame - segment.start_frame + 1
                if duration >= self.config.segments.min_action_frames:
                    self.debug_events.append(
                        {
                            "event_type": "segment_kept",
                            "scope": segment.hand_id,
                            "label": segment.label,
                            "start_frame": segment.start_frame,
                            "end_frame": segment.end_frame,
                            "reason": f"kept because duration {duration} >= min_action_frames {self.config.segments.min_action_frames}",
                        }
                    )
                    segments.append(segment)
                else:
                    self.debug_events.append(
                        {
                            "event_type": "segment_suppressed",
                            "scope": segment.hand_id,
                            "label": segment.label,
                            "start_frame": segment.start_frame,
                            "end_frame": segment.end_frame,
                            "reason": f"suppressed because duration {duration} < min_action_frames {self.config.segments.min_action_frames}",
                        }
                    )
        return sorted(segments, key=lambda segment: (segment.hand_id, segment.start_frame))

    def consolidate_relations(self, relation_records: Sequence[Dict[str, object]]) -> List[RelationSegment]:
        ordered = sorted(relation_records, key=lambda item: int(item["frame_index"]))
        candidates: List[RelationSegment] = []
        current: RelationSegment | None = None
        confidences: List[float] = []

        for record in ordered:
            label = str(record["relation_label"])
            if self.config.segments.suppress_idle and label == self.config.relations.idle_label:
                if current is not None:
                    current.confidence = sum(confidences) / max(len(confidences), 1)
                    candidates.append(current)
                    current = None
                    confidences = []
                continue

            frame_index = int(record["frame_index"])
            timestamp = float(record["timestamp"])
            confidence = float(record["relation_confidence"])
            if current and current.relation_label == label and frame_index <= current.end_frame + self.config.segments.relation_merge_gap_frames:
                current.end_frame = frame_index
                current.end_time = timestamp
                confidences.append(confidence)
            else:
                if current is not None:
                    current.confidence = sum(confidences) / max(len(confidences), 1)
                    candidates.append(current)
                current = RelationSegment(
                    relation_label=label,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    start_time=timestamp,
                    end_time=timestamp,
                    confidence=confidence,
                    notes=str(record.get("debug_explanation", "")),
                )
                confidences = [confidence]

        if current is not None:
            current.confidence = sum(confidences) / max(len(confidences), 1)
            candidates.append(current)

        return [
            segment
            for segment in self._merge_relation_segments(candidates)
            if (segment.end_frame - segment.start_frame + 1) >= self.config.segments.relation_min_frames
        ]

    def _build_action_candidates(self, ordered: Sequence[Dict[str, object]]) -> List[ActionSegment]:
        candidates: List[ActionSegment] = []
        current: ActionSegment | None = None
        confidences: List[float] = []

        for record in ordered:
            label = str(record["action_label"])
            if self.config.segments.suppress_idle and label == self.config.gestures.idle_label:
                if current is not None:
                    current.confidence = sum(confidences) / max(len(confidences), 1)
                    candidates.append(current)
                    current = None
                    confidences = []
                continue

            frame_index = int(record["frame_index"])
            timestamp = float(record["timestamp"])
            confidence = float(record["action_confidence"])
            if current and current.label == label and frame_index <= current.end_frame + self.config.segments.max_merge_gap_frames:
                current.end_frame = frame_index
                current.end_time = timestamp
                confidences.append(confidence)
            else:
                if current is not None:
                    current.confidence = sum(confidences) / max(len(confidences), 1)
                    candidates.append(current)
                current = ActionSegment(
                    hand_id=str(record["hand_id"]),
                    handedness=str(record["handedness"]),
                    label=label,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    start_time=timestamp,
                    end_time=timestamp,
                    confidence=confidence,
                    notes=str(record.get("debug_explanation", "")),
                )
                confidences = [confidence]

        if current is not None:
            current.confidence = sum(confidences) / max(len(confidences), 1)
            candidates.append(current)

        return candidates

    def _merge_action_segments(self, segments: Sequence[ActionSegment]) -> List[ActionSegment]:
        if not segments:
            return []
        merged: List[ActionSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if (
                previous.hand_id == segment.hand_id
                and previous.label == segment.label
                and (segment.start_frame - previous.end_frame) <= self.config.segments.max_merge_gap_frames
            ):
                duration_prev = previous.end_frame - previous.start_frame + 1
                duration_curr = segment.end_frame - segment.start_frame + 1
                total = duration_prev + duration_curr
                previous.end_frame = segment.end_frame
                previous.end_time = segment.end_time
                previous.confidence = (
                    (previous.confidence * duration_prev) + (segment.confidence * duration_curr)
                ) / max(total, 1)
                previous.notes = "; ".join(filter(None, [previous.notes, segment.notes]))
                self.debug_events.append(
                    {
                        "event_type": "segment_merged",
                        "scope": previous.hand_id,
                        "label": previous.label,
                        "start_frame": previous.start_frame,
                        "end_frame": previous.end_frame,
                        "reason": f"merged because label matched and gap <= {self.config.segments.max_merge_gap_frames} frames",
                    }
                )
            else:
                merged.append(segment)
        return merged

    def _merge_relation_segments(self, segments: Sequence[RelationSegment]) -> List[RelationSegment]:
        if not segments:
            return []
        merged: List[RelationSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if (
                previous.relation_label == segment.relation_label
                and (segment.start_frame - previous.end_frame) <= self.config.segments.relation_merge_gap_frames
            ):
                duration_prev = previous.end_frame - previous.start_frame + 1
                duration_curr = segment.end_frame - segment.start_frame + 1
                total = duration_prev + duration_curr
                previous.end_frame = segment.end_frame
                previous.end_time = segment.end_time
                previous.confidence = (
                    (previous.confidence * duration_prev) + (segment.confidence * duration_curr)
                ) / max(total, 1)
                previous.notes = "; ".join(filter(None, [previous.notes, segment.notes]))
                self.debug_events.append(
                    {
                        "event_type": "relation_merged",
                        "scope": "inter_hand",
                        "label": previous.relation_label,
                        "start_frame": previous.start_frame,
                        "end_frame": previous.end_frame,
                        "reason": f"merged because relation label matched and gap <= {self.config.segments.relation_merge_gap_frames} frames",
                    }
                )
            else:
                merged.append(segment)
        return merged
