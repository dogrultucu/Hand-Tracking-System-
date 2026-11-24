from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from config import AppConfig
from segments import ActionSegment


@dataclass
class ActionPhrase:
    phrase_scope: str
    phrase_label: str
    component_actions: List[str]
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    notes: str = ""

    def to_record(self) -> Dict[str, object]:
        return {
            "phrase_scope": self.phrase_scope,
            "phrase_label": self.phrase_label,
            "component_actions": "|".join(self.component_actions),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class _GrammarRule:
    phrase_label: str
    actions: tuple[str, ...]


class ActionGrammarAnalyzer:
    """Matches deterministic high-level phrases over stabilized low-level action segments."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.debug_events: List[Dict[str, object]] = []
        self.rules = [
            _GrammarRule("manipulate_and_insert", ("grab", "turn", "push")),
            _GrammarRule("twist_and_extract", ("grab", "hold", "pull + turn")),
            _GrammarRule("reposition_and_reengage", ("release + push", "grab")),
            _GrammarRule("reposition_and_reengage", ("release + push", "move + hold", "grab")),
        ]

    def analyze(self, action_segments: Sequence[ActionSegment]) -> List[ActionPhrase]:
        self.debug_events = []
        grouped: Dict[str, List[ActionSegment]] = {}
        for segment in action_segments:
            grouped.setdefault(segment.hand_id, []).append(segment)

        phrases: List[ActionPhrase] = []
        for hand_id, segments in grouped.items():
            ordered = sorted(segments, key=lambda item: item.start_frame)
            used_spans: List[tuple[int, int]] = []
            for rule in self.rules:
                for start_index in range(len(ordered)):
                    match = self._match_rule(rule, ordered, start_index)
                    if match is None:
                        continue
                    start_seg, end_seg, components, confidence = match
                    overlap = any(
                        not (end_seg.end_frame < used_start or start_seg.start_frame > used_end)
                        for used_start, used_end in used_spans
                    )
                    if overlap or confidence < self.config.grammar.min_phrase_confidence:
                        self.debug_events.append(
                            {
                                "event_type": "grammar_rejected",
                                "scope": hand_id,
                                "label": rule.phrase_label,
                                "start_frame": start_seg.start_frame,
                                "end_frame": end_seg.end_frame,
                                "reason": (
                                    "rejected because overlap was detected"
                                    if overlap
                                    else f"rejected because mean component confidence {confidence:.2f} < min_phrase_confidence {self.config.grammar.min_phrase_confidence:.2f}"
                                ),
                            }
                        )
                        continue
                    phrases.append(
                        ActionPhrase(
                            phrase_scope=hand_id,
                            phrase_label=rule.phrase_label,
                            component_actions=[segment.label for segment in components],
                            start_frame=start_seg.start_frame,
                            end_frame=end_seg.end_frame,
                            start_time=start_seg.start_time,
                            end_time=end_seg.end_time,
                            confidence=confidence,
                            notes=(
                                f"matched rule {' -> '.join(rule.actions)} because component actions were clean within "
                                f"{self.config.grammar.max_gap_frames} frames"
                            ),
                        )
                    )
                    used_spans.append((start_seg.start_frame, end_seg.end_frame))
                    self.debug_events.append(
                        {
                            "event_type": "grammar_matched",
                            "scope": hand_id,
                            "label": rule.phrase_label,
                            "start_frame": start_seg.start_frame,
                            "end_frame": end_seg.end_frame,
                            "reason": f"matched rule {' -> '.join(rule.actions)} with mean confidence {confidence:.2f}",
                        }
                    )
        return sorted(phrases, key=lambda phrase: (phrase.phrase_scope, phrase.start_frame))

    def _match_rule(
        self,
        rule: _GrammarRule,
        segments: Sequence[ActionSegment],
        start_index: int,
    ) -> tuple[ActionSegment, ActionSegment, List[ActionSegment], float] | None:
        components: List[ActionSegment] = []
        cursor = start_index
        last_end_frame: int | None = None
        for action_label in rule.actions:
            while cursor < len(segments) and segments[cursor].label != action_label:
                cursor += 1
            if cursor >= len(segments):
                self.debug_events.append(
                    {
                        "event_type": "grammar_rejected",
                        "scope": segments[start_index].hand_id if segments else "unknown",
                        "label": rule.phrase_label,
                        "start_frame": segments[start_index].start_frame if segments else -1,
                        "end_frame": segments[start_index].end_frame if segments else -1,
                        "reason": f"rejected because component action '{action_label}' was not found in sequence order",
                    }
                )
                return None
            current = segments[cursor]
            if current.confidence < self.config.segments.phrase_min_component_confidence:
                self.debug_events.append(
                    {
                        "event_type": "grammar_rejected",
                        "scope": current.hand_id,
                        "label": rule.phrase_label,
                        "start_frame": current.start_frame,
                        "end_frame": current.end_frame,
                        "reason": (
                            f"rejected because component '{action_label}' confidence {current.confidence:.2f} "
                            f"< phrase_min_component_confidence {self.config.segments.phrase_min_component_confidence:.2f}"
                        ),
                    }
                )
                return None
            if last_end_frame is not None:
                gap = current.start_frame - last_end_frame
                if gap > self.config.grammar.max_gap_frames:
                    self.debug_events.append(
                        {
                            "event_type": "grammar_rejected",
                            "scope": current.hand_id,
                            "label": rule.phrase_label,
                            "start_frame": current.start_frame,
                            "end_frame": current.end_frame,
                            "reason": f"rejected because time gap {gap} exceeded max_gap_frames {self.config.grammar.max_gap_frames}",
                        }
                    )
                    return None
                if gap < -self.config.grammar.allow_overlap_margin_frames:
                    self.debug_events.append(
                        {
                            "event_type": "grammar_rejected",
                            "scope": current.hand_id,
                            "label": rule.phrase_label,
                            "start_frame": current.start_frame,
                            "end_frame": current.end_frame,
                            "reason": (
                                f"rejected because overlap gap {gap} exceeded allow_overlap_margin_frames "
                                f"{self.config.grammar.allow_overlap_margin_frames}"
                            ),
                        }
                    )
                    return None
            components.append(current)
            last_end_frame = current.end_frame
            cursor += 1

        mean_confidence = sum(segment.confidence for segment in components) / max(len(components), 1)
        return components[0], components[-1], components, mean_confidence
