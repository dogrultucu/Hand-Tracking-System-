from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from ui_components import (
    build_filters,
    filter_timeline_df,
    render_feature_panel,
    render_metadata_panel,
    render_review_note_form,
    render_segment_inspector,
    render_threshold_panel,
    render_uncertainty_panel,
    render_video_panel,
)
from ui_state import (
    append_review_note,
    discover_review_runs,
    load_run_data,
    save_review_notes,
)
from ui_timeline import build_timeline_dataframe, make_timeline_figure


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive local review UI for hand analysis results.")
    parser.add_argument("--results_dir", required=True, help="Single run directory or batch results root.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    st.set_page_config(page_title="Hand Analysis Review", layout="wide")
    st.title("Hand Analysis Review Workspace")

    run_dirs = discover_review_runs(args.results_dir)
    if not run_dirs:
        st.error("No reviewable runs found in the provided directory.")
        return

    selected_run_dir = st.sidebar.selectbox("Run", [str(path) for path in run_dirs])
    run_data = load_run_data(selected_run_dir)

    sibling_runs = {
        path.name: load_run_data(path)
        for path in run_dirs
        if str(path) != selected_run_dir
    }
    compare_selection = st.sidebar.multiselect("Compare Runs", list(sibling_runs.keys()))
    compare_runs = {name: sibling_runs[name] for name in compare_selection}

    filters = build_filters(run_data)
    video_mode = st.sidebar.radio("Video Source", ["analytic_overlay", "ghost_diagram", "original"], horizontal=False)

    max_time = float(run_data.run_metadata.get("duration_seconds", 0.0))
    if max_time <= 0.0 and not run_data.frame_df.empty:
        max_time = float(run_data.frame_df["timestamp"].max())
    selected_time = st.sidebar.slider("Inspection Time (s)", 0.0, max(max_time, 0.01), 0.0, 0.01)

    timeline_df = build_timeline_dataframe(
        run_data.frame_df,
        run_data.primitive_df,
        run_data.action_segments_df,
        run_data.phrase_df,
        run_data.relation_segments_df,
        left_hex=str(run_data.threshold_report.get("visualization", {}).get("left_hand_hex", "#06065D")),
        right_hex=str(run_data.threshold_report.get("visualization", {}).get("right_hand_hex", "#ED0101")),
    )
    filtered_timeline_df = filter_timeline_df(timeline_df, filters, run_data.debug_events_df)
    figure = make_timeline_figure(filtered_timeline_df)

    top_left, top_right = st.columns([1.5, 1.0], gap="large")
    with top_left:
        render_video_panel(run_data, video_mode)
    with top_right:
        render_metadata_panel(run_data)
        render_uncertainty_panel(run_data)

    st.subheader("Multi-Track Timeline")
    st.plotly_chart(figure, use_container_width=True)

    selectable_segments = filtered_timeline_df.sort_values(["start_time", "track"]).copy()
    if selectable_segments.empty:
        segment_options = []
    else:
        segment_options = selectable_segments["segment_key"].tolist()
    selected_segment_key = st.selectbox(
        "Select Segment",
        options=segment_options,
        format_func=lambda key: _format_segment_option(selectable_segments, key),
    )
    selected_segment = None
    if selected_segment_key:
        selected_rows = selectable_segments[selectable_segments["segment_key"] == selected_segment_key]
        if not selected_rows.empty:
            selected_segment = selected_rows.iloc[0].to_dict()
            selected_time = float(selected_segment["start_time"])

    inspector_col, feature_col = st.columns([1.0, 1.2], gap="large")
    with inspector_col:
        review_note_matches = pd.DataFrame()
        if selected_segment is not None and not run_data.review_notes_df.empty:
            review_note_matches = run_data.review_notes_df[
                run_data.review_notes_df["segment_key"] == selected_segment["segment_key"]
            ]
        render_segment_inspector(selected_segment, run_data, review_note_matches)
        review_note_payload = render_review_note_form(selected_segment)
        if review_note_payload and selected_segment is not None:
            review_notes_df = append_review_note(
                run_data.review_notes_df,
                segment_type=str(selected_segment["segment_type"]),
                segment_key=str(selected_segment["segment_key"]),
                verdict=review_note_payload["verdict"],
                alternate_label=review_note_payload["alternate_label"],
                note=review_note_payload["note"],
                selected_time=selected_time,
            )
            save_review_notes(run_data.run_dir, review_notes_df)
            st.success("Review note saved.")

    with feature_col:
        signal_candidates = [
            "openness_score",
            "motion_magnitude",
            "rotation_velocity",
            "thumb_index_distance",
            "tracker_stability",
            "handedness_score",
            "action_confidence",
            "visibility_score",
        ]
        default_signals = ["openness_score", "motion_magnitude", "rotation_velocity", "action_confidence"]
        selected_signals = st.multiselect("Signals", signal_candidates, default=default_signals)
        render_feature_panel(run_data.frame_df, selected_signals, str(filters["hand"]), selected_time)

    lower_left, lower_right = st.columns([1.2, 1.0], gap="large")
    with lower_left:
        st.subheader("Filtered Segment Table")
        st.dataframe(
            selectable_segments[
                ["track", "label", "start_time", "end_time", "confidence", "segment_type", "scope"]
            ] if not selectable_segments.empty else pd.DataFrame(),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Debug Events")
        st.dataframe(run_data.debug_events_df, use_container_width=True, hide_index=True)

    with lower_right:
        render_threshold_panel(run_data, compare_runs)
        with st.expander("Evaluation Summary", expanded=False):
            st.json(run_data.evaluation_summary or {"message": "No evaluation summary available"})
        with st.expander("Review Notes", expanded=False):
            st.dataframe(run_data.review_notes_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Review Notes JSON",
            data=run_data.review_notes_df.to_json(orient="records", indent=2),
            file_name="review_notes.json",
            mime="application/json",
        )


def _format_segment_option(selectable_segments: pd.DataFrame, key: str) -> str:
    if not key:
        return ""
    row = selectable_segments[selectable_segments["segment_key"] == key].iloc[0]
    return (
        f"{row['track']} | {row['label']} | "
        f"{float(row['start_time']):.2f}-{float(row['end_time']):.2f}s | "
        f"{float(row['confidence']):.2f}"
    )


if __name__ == "__main__":
    main()
