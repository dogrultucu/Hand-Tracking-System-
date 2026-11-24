from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from annotations import AnnotationRecord
from config import AppConfig


@dataclass
class EvaluationResult:
    summary: Dict[str, object]
    metrics_table: pd.DataFrame


class Evaluator:
    """Evaluates predicted actions, phrases, and relations against optional reference annotations."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def evaluate(
        self,
        frame_df: pd.DataFrame,
        action_segments_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_segments_df: pd.DataFrame,
        annotations: Sequence[AnnotationRecord] | None = None,
        tracking_summary: Dict[str, int] | None = None,
    ) -> EvaluationResult:
        tracking_summary = tracking_summary or {}
        summary = self._descriptive_summary(frame_df, action_segments_df, phrase_df, relation_segments_df, tracking_summary)
        metrics_frames: List[pd.DataFrame] = []

        if annotations:
            grouped = defaultdict(list)
            for record in annotations:
                grouped[record.target_type].append(record)

            action_metrics, action_confusion = self._evaluate_segments(
                predicted_df=action_segments_df,
                reference_records=grouped.get("action", []),
                label_col="action_label" if "action_label" in action_segments_df.columns else "label",
                scope_col="hand_id",
                iou_threshold=self.config.evaluation.segment_iou_threshold,
                metric_scope="action",
            )
            phrase_metrics, phrase_confusion = self._evaluate_segments(
                predicted_df=phrase_df,
                reference_records=grouped.get("phrase", []),
                label_col="phrase_label",
                scope_col="phrase_scope",
                iou_threshold=self.config.evaluation.phrase_iou_threshold,
                metric_scope="phrase",
            )
            relation_metrics, relation_confusion = self._evaluate_segments(
                predicted_df=relation_segments_df,
                reference_records=grouped.get("relation", []),
                label_col="relation_label",
                scope_col=None,
                iou_threshold=self.config.evaluation.relation_iou_threshold,
                metric_scope="relation",
            )
            metrics_frames.extend([action_metrics, phrase_metrics, relation_metrics])
            summary["action_confusion"] = action_confusion
            summary["phrase_confusion"] = phrase_confusion
            summary["relation_confusion"] = relation_confusion
            summary["ground_truth_available"] = True
        else:
            summary["ground_truth_available"] = False

        metrics_table = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
        return EvaluationResult(summary=summary, metrics_table=metrics_table)

    def _descriptive_summary(
        self,
        frame_df: pd.DataFrame,
        action_segments_df: pd.DataFrame,
        phrase_df: pd.DataFrame,
        relation_segments_df: pd.DataFrame,
        tracking_summary: Dict[str, int],
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "total_frames_with_hands": int(frame_df["frame_index"].nunique()) if not frame_df.empty else 0,
            "detected_hands": sorted(frame_df["hand_id"].unique().tolist()) if not frame_df.empty else [],
            "action_counts": action_segments_df["action_label"].value_counts().to_dict() if not action_segments_df.empty else {},
            "phrase_counts": phrase_df["phrase_label"].value_counts().to_dict() if not phrase_df.empty else {},
            "relation_counts": relation_segments_df["relation_label"].value_counts().to_dict() if not relation_segments_df.empty else {},
            "avg_action_duration_seconds": float(
                ((action_segments_df["end_time"] - action_segments_df["start_time"]).mean()) if not action_segments_df.empty else 0.0
            ),
            "avg_action_confidence_by_class": (
                action_segments_df.groupby("action_label")["confidence"].mean().round(4).to_dict()
                if not action_segments_df.empty and "confidence" in action_segments_df.columns
                else {}
            ),
            "primitive_uncertain_frequency": float(
                frame_df["primitive_labels"].fillna("").str.contains("uncertain_motion").mean()
                if not frame_df.empty and "primitive_labels" in frame_df.columns
                else 0.0
            ),
            "tracking_summary": tracking_summary,
            "warning_flags": self._warning_flags(frame_df, action_segments_df, tracking_summary),
        }
        return summary

    def _warning_flags(
        self,
        frame_df: pd.DataFrame,
        action_segments_df: pd.DataFrame,
        tracking_summary: Dict[str, int],
    ) -> List[str]:
        warnings: List[str] = []
        if tracking_summary.get("suspected_identity_switches", 0) > 0:
            warnings.append("suspected_identity_switches")
        if not frame_df.empty and frame_df["tracker_stability"].mean() < 0.55:
            warnings.append("low_tracker_stability")
        if not frame_df.empty and frame_df["visibility_score"].mean() < 0.5:
            warnings.append("low_visibility")
        if action_segments_df.empty:
            warnings.append("no_actions_detected")
        return warnings

    def _evaluate_segments(
        self,
        predicted_df: pd.DataFrame,
        reference_records: Sequence[AnnotationRecord],
        label_col: str,
        scope_col: str | None,
        iou_threshold: float,
        metric_scope: str,
    ) -> tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        if predicted_df.empty and not reference_records:
            return pd.DataFrame(), {}

        predictions = predicted_df.to_dict(orient="records")
        references = [record.to_record() for record in reference_records]
        labels = sorted(
            set(item.get(label_col, item.get("label")) for item in predictions)
            | set(record["label"] for record in references)
        )
        metrics: List[Dict[str, object]] = []
        confusion: Dict[str, Counter[str]] = defaultdict(Counter)

        for label in labels:
            pred_items = [item for item in predictions if item.get(label_col, item.get("label")) == label]
            ref_items = [item for item in references if item["label"] == label]
            matches, unmatched_preds, unmatched_refs = self._match_segments(pred_items, ref_items, scope_col, iou_threshold)
            precision = len(matches) / max(len(pred_items), 1)
            recall = len(matches) / max(len(ref_items), 1)
            f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
            avg_iou = sum(match["iou"] for match in matches) / max(len(matches), 1)
            avg_conf = (
                sum(float(item.get("confidence", 0.0)) for item in pred_items) / max(len(pred_items), 1)
                if pred_items
                else 0.0
            )
            metrics.append(
                {
                    "metric_scope": metric_scope,
                    "label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "avg_overlap_iou": avg_iou,
                    "avg_confidence": avg_conf,
                    "pred_count": len(pred_items),
                    "ref_count": len(ref_items),
                    "matched_count": len(matches),
                }
            )
            for mismatch in unmatched_preds:
                confusion[label]["false_positive"] += 1
            for mismatch in unmatched_refs:
                confusion[label]["false_negative"] += 1

        return pd.DataFrame(metrics), {label: dict(counter) for label, counter in confusion.items()}

    def _match_segments(
        self,
        predictions: Sequence[Dict[str, object]],
        references: Sequence[Dict[str, object]],
        scope_col: str | None,
        iou_threshold: float,
    ) -> tuple[List[Dict[str, float]], List[Dict[str, object]], List[Dict[str, object]]]:
        matches: List[Dict[str, float]] = []
        unmatched_preds = list(predictions)
        unmatched_refs = list(references)

        for prediction in predictions:
            best_ref = None
            best_iou = 0.0
            for reference in unmatched_refs:
                if scope_col is not None:
                    prediction_scope = {str(prediction.get(scope_col, "")), str(prediction.get("handedness", ""))}
                    if str(reference.get("scope", "")) not in prediction_scope:
                        continue
                iou = self._temporal_iou(
                    float(prediction.get("start_time", 0.0)),
                    float(prediction.get("end_time", 0.0)),
                    float(reference.get("start_time", 0.0)),
                    float(reference.get("end_time", 0.0)),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_ref = reference
            if best_ref is not None and best_iou >= iou_threshold:
                matches.append({"iou": best_iou})
                unmatched_preds.remove(prediction)
                unmatched_refs.remove(best_ref)
        return matches, unmatched_preds, unmatched_refs

    @staticmethod
    def _temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
        union = max(end_a, end_b) - min(start_a, start_b)
        if union <= 0.0:
            return 0.0
        return inter / union
