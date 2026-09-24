"""Feature imports and deliberate, bounded experiment execution."""

import json

import pandas as pd
import streamlit as st

from seismic.enrichment import (
    GEOLOGY_FEATURES, WAVEFORM_FEATURES, attach_features, geology_features, waveform_features,
)
from seismic.modeling import catalogue_fingerprint
from seismic.paths import ARTIFACTS


def feature_workbench(events):
    st.caption("Uses the full loaded catalogue. Only user-supplied measurements are joined, by event ID.")
    st.markdown(
        "Waveform CSV: `event_id,station,channel,timestamp,amplitude,unit,source,available_at`. "
        "Use uniformly sampled, contiguous traces with at least eight samples. "
        "Summaries are demeaned RMS, peak amplitude and dominant FFT frequency; "
        "they are not ground-motion hazard estimates."
    )
    waveforms = st.file_uploader("Waveform samples", type="csv", key="waveforms")
    st.markdown(
        "Geology CSV: `event_id,source,available_at` plus `geo_fault_distance_km`, "
        "`geo_vs30_m_s` and/or `geo_elevation_m`. All timestamps require explicit offsets. "
        "Fault distances and geological values must come from your documented source."
    )
    geology = st.file_uploader("Geological features", type="csv", key="geology")
    if waveforms is None and geology is None:
        st.info("No waveform or geological measurements are bundled. Supply source data to enable enrichment.")
        return
    try:
        enriched = events.copy()
        for upload, extractor in ((waveforms, waveform_features), (geology, geology_features)):
            if upload is not None:
                upload.seek(0)
                raw = pd.read_csv(upload, dtype={"event_id": "string"})
                enriched = attach_features(enriched, extractor(raw))
        columns = [c for c in (*WAVEFORM_FEATURES, *GEOLOGY_FEATURES) if c in enriched]
        st.dataframe(enriched[["event_id", "date_time", *columns]], use_container_width=True)
        complete = enriched[columns].notna().all(axis=1).sum()
        st.caption(f"Events with every selected feature: {complete} / {len(enriched)}")
        st.download_button("Download enriched catalogue", enriched.to_csv(index=False).encode("utf-8"),
                           "enriched_events.csv", "text/csv")
        st.info("Load the downloaded catalogue in the sidebar to use its features in Research Experiments.")
    except (ValueError, OSError) as exc:
        st.error(f"Feature import rejected: {exc}")


def research_experiments(events):
    st.caption(
        "Uses the full loaded catalogue. Candidates are ranked by validation RMSE; "
        "only the selected network is evaluated on the final test block."
    )
    st.warning(
        "The bundled 70-event holdout has already been inspected. Experiments on it "
        "demonstrate the workflow, not independent confirmation of forecast skill."
    )
    architectures = st.multiselect("Candidate architectures", ["lstm", "gru", "cnn"],
                                   default=["lstm", "gru", "cnn"])
    units = st.multiselect("Units / filters", [8, 16, 32], default=[8])
    rate = st.selectbox("Learning rate", [0.001, 0.003, 0.0003])
    epochs = st.number_input("Maximum epochs per candidate", min_value=1, max_value=200, value=50)
    available = [c for c in (*WAVEFORM_FEATURES, *GEOLOGY_FEATURES) if c in events]
    features = st.multiselect("Additional measured features", available)
    delay = st.number_input("Forecast issue delay after previous event (seconds)",
                            min_value=0.0, max_value=86400.0, value=0.0)
    st.caption(
        "Features must be available by issue time, and the target event must "
        "occur afterwards. Ineligible windows are excluded."
    )
    output = ARTIFACTS / "experiment"
    if st.button("Run validation experiment"):
        from seismic.experiments import experiment_grid, run_experiments

        try:
            configurations = experiment_grid(tuple(architectures), tuple(units), (rate,))
            with st.spinner("Training candidates and comparing validation results..."):
                run_experiments(events, output, configurations, epochs=int(epochs),
                                extra_features=tuple(features), forecast_delay_seconds=delay)
        except (ValueError, OSError) as exc:
            st.error(f"Experiment failed: {exc}")
    if not (output / "selection.json").exists():
        st.info("No completed experiment is available. Training starts only when you press the button.")
        return
    try:
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        used_features = tuple(selection.get("extra_features", []))
        if selection["dataset_sha256"] != catalogue_fingerprint(events, used_features):
            st.info("Saved experiment belongs to a different catalogue.")
            return
        st.subheader("Validation comparison")
        st.dataframe(pd.read_csv(output / "validation_comparison.csv"), hide_index=True)
        st.subheader("Selected network versus baselines — test set")
        st.dataframe(pd.read_csv(output / "selected" / "metrics.csv"), hide_index=True)
        with st.expander("Selection protocol"):
            st.json(selection)
        st.image(str(output / "selected" / "actual_vs_predicted.png"))
        st.image(str(output / "selected" / "training_loss.png"))
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved experiment cannot be displayed for this catalogue: {exc}")
