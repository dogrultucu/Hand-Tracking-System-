from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import json

from ui_state import ReviewRunData, flatten_thresholds


def render_metadata_panel(run_data: ReviewRunData) -> None:
    metadata = run_data.run_metadata
    eval_summary = run_data.evaluation_summary
    st.subheader("Run Metadata")
    st.markdown(f"**Video:** `{metadata.get('input_video_name', run_data.run_dir.name)}`")
    st.markdown(f"**Preset:** `{metadata.get('profile_name', run_data.threshold_report.get('runtime', {}).get('profile_name', 'unknown'))}`")
    st.markdown(f"**Processed:** `{metadata.get('processed_at_utc', 'unknown')}`")
    st.markdown(f"**Duration:** `{metadata.get('duration_seconds', 0):.2f}s`")
    st.markdown(f"**Results Dir:** `{run_data.run_dir}`")
    warnings = metadata.get("warnings", eval_summary.get("warning_flags", []))
    st.markdown(f"**Warnings:** `{', '.join(warnings) if warnings else 'none'}`")
    if "output_paths" in metadata:
        with st.expander("Output Paths", expanded=False):
            for key, value in metadata["output_paths"].items():
                st.code(f"{key}: {value}")


def render_video_panel(run_data: ReviewRunData, selected_video: str) -> None:
    metadata = run_data.run_metadata
    video_map = {
        "original": Path(str(metadata.get("input_video_path", ""))),
        "analytic_overlay": run_data.run_dir / "annotated.mp4",
        "ghost_diagram": run_data.run_dir / "ghost_diagram.mp4",
    }
    path = video_map[selected_video]
    st.subheader("Video Review")
    if path.exists():
        st.video(str(path))
    else:
        st.info(f"{selected_video} video is not available for this run.")


def render_threshold_panel(run_data: ReviewRunData, compare_runs: Dict[str, ReviewRunData]) -> None:
    st.subheader("Threshold Review")
    thresholds_df = pd.DataFrame(flatten_thresholds(run_data.threshold_report))
    if thresholds_df.empty:
        st.info("No threshold report found.")
        return
    st.dataframe(thresholds_df, use_container_width=True, hide_index=True)
    if compare_runs:
        compare_rows = []
        base_map = {f"{row['category']}.{row['name']}": row["value"] for row in thresholds_df.to_dict(orient="records")}
        for compare_name, compare_data in compare_runs.items():
            compare_df = pd.DataFrame(flatten_thresholds(compare_data.threshold_report))
            for row in compare_df.to_dict(orient="records"):
                key = f"{row['category']}.{row['name']}"
                compare_rows.append(
                    {
                        "parameter": key,
                        "current_value": base_map.get(key),
                        "compare_run": compare_name,
                        "compare_value": row["value"],
                    }
                )
        if compare_rows:
            with st.expander("Compare Runs", expanded=False):
                st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Threshold JSON",
        data=json.dumps(run_data.threshold_report, indent=2),
        file_name="thresholds_used.json",
        mime="application/json",
    )


def render_feature_panel(frame_df: pd.DataFrame, selected_signals: List[str], selected_hand: str, selected_time: float) -> None:
    st.subheader("Feature / Debug Signals")
    if frame_df.empty:
        st.info("No frame data available.")
        return
    filtered = frame_df.copy()
    if selected_hand != "all":
        filtered = filtered[filtered["handedness"] == selected_hand]
    fig = go.Figure()
    for signal in selected_signals:
        if signal not in filtered.columns:
            continue
        for hand_id, hand_df in filtered.groupby("hand_id"):
            fig.add_trace(
                go.Scatter(
                    x=hand_df["timestamp"],
                    y=hand_df[signal],
                    mode="lines",
                    name=f"{hand_id}:{signal}",
                )
            )
    fig.add_vline(x=selected_time, line_dash="dot", line_color="#222222")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def render_segment_inspector(
    selected_segment: Dict[str, object] | None,
    run_data: ReviewRunData,
    review_note_matches: pd.DataFrame,
) -> None:
    st.subheader("Segment Inspector")
    if not selected_segment:
        st.info("Select a segment to inspect.")
        return
    st.markdown(f"**Label:** `{selected_segment.get('label', '')}`")
    st.markdown(f"**Scope:** `{selected_segment.get('scope', '')}`")
    st.markdown(f"**Start:** `{float(selected_segment.get('start_time', 0.0)):.2f}s`")
    st.markdown(f"**End:** `{float(selected_segment.get('end_time', 0.0)):.2f}s`")
    st.markdown(
        f"**Duration:** `{float(selected_segment.get('end_time', 0.0)) - float(selected_segment.get('start_time', 0.0)):.2f}s`"
    )
    st.markdown(f"**Confidence:** `{float(selected_segment.get('confidence', 0.0)):.2f}`")
    if selected_segment.get("segment_type") == "action":
        overlap_primitives = run_data.timeline_summary_df[
            (run_data.timeline_summary_df["hand_id"] == selected_segment.get("hand_id"))
            & (run_data.timeline_summary_df["timestamp"] >= float(selected_segment.get("start_time", 0.0)))
            & (run_data.timeline_summary_df["timestamp"] <= float(selected_segment.get("end_time", 0.0)))
        ]["primitive_labels"].value_counts()
        st.markdown(f"**Linked primitives:** `{', '.join(overlap_primitives.head(5).index.tolist())}`")
    if selected_segment.get("segment_type") in {"action", "phrase"}:
        overlap_phrases = run_data.phrase_df[
            (run_data.phrase_df["start_time"] <= float(selected_segment.get("end_time", 0.0)))
            & (run_data.phrase_df["end_time"] >= float(selected_segment.get("start_time", 0.0)))
        ]
        if not overlap_phrases.empty:
            st.markdown(f"**Linked grammar phrase:** `{', '.join(overlap_phrases['phrase_label'].astype(str).tolist())}`")
    overlap_relations = run_data.relation_segments_df[
        (run_data.relation_segments_df["start_time"] <= float(selected_segment.get("end_time", 0.0)))
        & (run_data.relation_segments_df["end_time"] >= float(selected_segment.get("start_time", 0.0)))
    ]
    if not overlap_relations.empty:
        st.markdown(f"**Linked relation state:** `{', '.join(overlap_relations['relation_label'].astype(str).tolist())}`")
    st.markdown(f"**Debug explanation:** {selected_segment.get('notes', 'n/a')}")
    snapshot = _find_snapshot(run_data.snapshots, str(selected_segment.get("label", "")))
    if snapshot is not None:
        st.image(str(snapshot), caption=snapshot.name, use_container_width=True)
    if not review_note_matches.empty:
        with st.expander("Existing Review Notes", expanded=False):
            st.dataframe(review_note_matches, use_container_width=True, hide_index=True)


def render_uncertainty_panel(run_data: ReviewRunData) -> None:
    st.subheader("Uncertainty Review")
    warnings = []
    if not run_data.action_segments_df.empty:
        low_actions = run_data.action_segments_df[run_data.action_segments_df["confidence"] < 0.6]
        warnings.append(("Low-confidence action segments", len(low_actions)))
    if not run_data.phrase_df.empty:
        low_phrases = run_data.phrase_df[run_data.phrase_df["confidence"] < 0.6]
        warnings.append(("Low-confidence grammar phrases", len(low_phrases)))
    if not run_data.relation_segments_df.empty:
        low_rel = run_data.relation_segments_df[run_data.relation_segments_df["confidence"] < 0.55]
        warnings.append(("Relation ambiguity markers", len(low_rel)))
    if not run_data.debug_events_df.empty:
        switch_rows = run_data.debug_events_df[run_data.debug_events_df["label"].astype(str).str.contains("switch", case=False, na=False)]
        warnings.append(("Identity instability markers", len(switch_rows)))
    if run_data.evaluation_summary.get("warning_flags"):
        warnings.append(("Run-level warning flags", len(run_data.evaluation_summary["warning_flags"])))
    if warnings:
        st.dataframe(pd.DataFrame(warnings, columns=["marker", "count"]), use_container_width=True, hide_index=True)
    else:
        st.info("No uncertainty markers were found for this run.")


def render_review_note_form(selected_segment: Dict[str, object] | None) -> Dict[str, str] | None:
    st.subheader("Annotation Assist")
    if not selected_segment:
        st.info("Select a segment before adding a review note.")
        return None
    with st.form("review_note_form"):
        verdict = st.selectbox("Review verdict", ["correct", "incorrect", "uncertain"])
        alternate_label = st.text_input("Suggested alternate label")
        note = st.text_area("Reviewer note")
        submitted = st.form_submit_button("Save Review Note")
    if submitted:
        return {
            "verdict": verdict,
            "alternate_label": alternate_label,
            "note": note,
        }
    return None


def build_filters(run_data: ReviewRunData) -> Dict[str, object]:
    hands = ["all", "Left", "Right"]
    action_options = ["all"] + sorted(run_data.action_segments_df.get("action_label", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    phrase_options = ["all"] + sorted(run_data.phrase_df.get("phrase_label", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    relation_options = ["all"] + sorted(run_data.relation_segments_df.get("relation_label", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    with st.sidebar:
        st.header("Filters")
        hand = st.selectbox("Hand", hands)
        action = st.selectbox("Action", action_options)
        phrase = st.selectbox("Grammar phrase", phrase_options)
        relation = st.selectbox("Relation", relation_options)
        confidence_range = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)
        uncertain_only = st.checkbox("Uncertain only")
        suspect_identity = st.checkbox("Identity instability only")
        occlusion_only = st.checkbox("Occlusion / low visibility only")
        search_text = st.text_input("Search labels / notes")
    return {
        "hand": hand,
        "action": action,
        "phrase": phrase,
        "relation": relation,
        "confidence_range": confidence_range,
        "uncertain_only": uncertain_only,
        "suspect_identity": suspect_identity,
        "occlusion_only": occlusion_only,
        "search_text": search_text.strip().lower(),
    }


def filter_timeline_df(timeline_df: pd.DataFrame, filters: Dict[str, object], debug_events_df: pd.DataFrame) -> pd.DataFrame:
    filtered = timeline_df.copy()
    if filtered.empty:
        return filtered
    if filters["hand"] != "all":
        filtered = filtered[filtered["scope"].astype(str).str.contains(str(filters["hand"]), na=False)]
    if filters["action"] != "all":
        filtered = filtered[(filtered["segment_type"] != "action") | (filtered["label"] == filters["action"])]
    if filters["phrase"] != "all":
        filtered = filtered[(filtered["segment_type"] != "phrase") | (filtered["label"] == filters["phrase"])]
    if filters["relation"] != "all":
        filtered = filtered[(filtered["segment_type"] != "relation") | (filtered["label"] == filters["relation"])]
    low, high = filters["confidence_range"]
    filtered = filtered[(filtered["confidence"] >= low) & (filtered["confidence"] <= high)]
    if filters["uncertain_only"]:
        filtered = filtered[(filtered["confidence"] < 0.65) | filtered["label"].astype(str).str.contains("uncertain", case=False, na=False)]
    if filters["search_text"]:
        token = str(filters["search_text"])
        filtered = filtered[
            filtered["label"].astype(str).str.lower().str.contains(token, na=False)
            | filtered["notes"].astype(str).str.lower().str.contains(token, na=False)
        ]
    if filters["suspect_identity"] and not debug_events_df.empty:
        suspect_scopes = debug_events_df[
            debug_events_df["reason"].astype(str).str.contains("switch|identity", case=False, na=False)
        ]["scope"].astype(str).unique().tolist()
        filtered = filtered[filtered["scope"].astype(str).isin(suspect_scopes + ["tracker", "inter_hand"])]
    if filters["occlusion_only"]:
        filtered = filtered[
            filtered["notes"].astype(str).str.contains("visibility|occlusion|uncertain", case=False, na=False)
            | filtered["label"].astype(str).str.contains("uncertain", case=False, na=False)
        ]
    return filtered


def _find_snapshot(snapshots: Iterable[Path], label: str) -> Path | None:
    for snapshot in snapshots:
        if label.replace(" ", "_").replace("+", "plus") in snapshot.name:
            return snapshot
    return next(iter(snapshots), None) if snapshots else None
