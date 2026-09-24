"""Small dashboard page renderers using shared, testable analytics."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from seismic.analysis import descriptive_statistics, event_sequences, strongest_events, temporal_summary
from seismic.modeling import catalogue_fingerprint
from seismic.paths import ARTIFACTS
from seismic.similarity import similar_events, subsequent_activity
from seismic.visualization import (
    category_chart, correlation_chart, distribution_chart, earthquake_map,
    frequency_chart, magnitude_depth_chart, style,
)

PAGES = ["Overview", "Earthquake Explorer", "Temporal Analysis", "Spatial Analysis",
         "Magnitude & Depth Analysis", "Similarity Finder", "Forecasting / Model Evaluation",
         "Data Quality", "Live Monitoring", "Feature Workbench", "Research Experiments", "About"]


def chart(figure):
    st.plotly_chart(figure, use_container_width=True)


def overview(data):
    cards = st.columns(4)
    for column, label, value in zip(
        cards, ["Recorded events", "Average magnitude", "Maximum magnitude", "Average depth (km)"],
        [str(len(data)), f"{data.mag.mean():.2f}", f"{data.mag.max():.1f}",
         f"{data.depth.mean():.1f}" if data.depth.notna().any() else "Unknown"],
    ):
        column.metric(label, value)
    strongest = data.loc[data.mag.idxmax()]
    st.caption(f"Strongest event: M {strongest.mag:.1f} · {strongest.location} · {strongest.date_time}")
    chart(frequency_chart(data))
    st.caption("Counts reflect catalogue coverage. First and last calendar bins may be partial.")
    st.subheader("Strongest recorded events")
    st.dataframe(strongest_events(data), use_container_width=True, hide_index=True)


def explorer(data):
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.download_button("Download filtered events", data.to_csv(index=False).encode("utf-8"),
                       "filtered_earthquakes.csv", "text/csv")
    st.subheader("Descriptive statistics")
    st.dataframe(descriptive_statistics(data), use_container_width=True)


def temporal(data):
    frequency = st.selectbox("Count interval", ["Daily", "Weekly", "Monthly"])
    chart(frequency_chart(data, {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[frequency]))
    window = st.selectbox("Rolling window (calendar days)", [7, 1, 30])
    rolling = temporal_summary(data, window)
    chart(style(px.line(rolling, x="date_time", y="rolling_events", markers=True,
                        title=f"Events in trailing {window} calendar days",
                        labels={"date_time": "Recorded day", "rolling_events": "Event count"})))
    chart(style(px.line(rolling, x="date_time", y="rolling_mean_magnitude", markers=True,
                        title=f"Event-weighted mean magnitude over {window} calendar days")))
    chart(style(px.histogram(event_sequences(data), x="hours_since_previous",
                             title="Time between consecutive recorded earthquakes",
                             labels={"hours_since_previous": "Hours since previous event"})))
    chart(category_chart(data, "month", "Events by calendar month"))
    if data.year.nunique() > 1:
        chart(frequency_chart(data, "YS"))
    else:
        st.caption("Only one year is represented; annual comparisons are unavailable.")
    st.caption(
        "Partial windows are included. Bursts are descriptive; no foreshock, "
        "aftershock or swarm classification is made."
    )


def spatial(data):
    figure = earthquake_map(data)
    if figure is None:
        st.info(
            "No usable event coordinates. Upload a CSV with latitude and longitude "
            "to enable mapping. City centroids are not earthquake epicentres."
        )
    else:
        mapped = data[["latitude", "longitude"]].notna().all(axis=1).sum()
        st.caption(f"Mapping {mapped} of {len(data)} filtered events. Basemap tiles require internet access.")
        chart(figure)


def magnitude_depth(data):
    left, right = st.columns(2)
    with left:
        chart(distribution_chart(data, "mag"))
        chart(category_chart(data, "magnitude_category", "Magnitude categories"))
    with right:
        chart(distribution_chart(data, "depth"))
        chart(category_chart(data, "depth_category", "Depth categories"))
    chart(magnitude_depth_chart(data))
    chart(correlation_chart(data))
    st.caption(
        "Depth: shallow <70 km; intermediate 70–300 km; deep >300 km. "
        "Correlation does not establish causation or forecast skill."
    )


def similarity(data, full_data):
    st.caption(
        "Historical similarity analysis, not earthquake prediction. Candidates "
        "come from the full loaded catalogue and must precede the selected event."
    )
    labels = {row.event_id: f"{row.date_time} | M {row.mag:.1f} | {row.location}"
              for row in data.itertuples()}
    selected = st.selectbox("Select event", data.event_id.tolist(), index=len(data) - 1,
                            format_func=lambda value: labels[value])
    count = st.slider("Similar events", 1, 10, 5)
    target = full_data.loc[full_data.event_id == selected].iloc[0]
    located = pd.notna(target.latitude) and pd.notna(target.longitude)
    geography = st.checkbox("Include geographic distance", disabled=not located, value=False)
    matches = similar_events(full_data, selected, count, use_geography=geography)
    if matches.empty:
        st.info("No eligible earlier events with the required measurements.")
        return
    columns = ["event_id", "date_time", "mag", "depth", "location", "distance", "similarity", "features_used"]
    if geography:
        columns.append("geographic_distance_km")
    st.dataframe(matches[columns], use_container_width=True, hide_index=True)
    st.caption(
        "Distance uses historical standard deviations for magnitude/depth; "
        "geographic distance is divided by 100 km. Similarity = 1/(1+distance), "
        "not a probability. Constant features use a scale of 1."
    )
    st.subheader("Subsequent recorded activity")
    st.caption(
        "Catalogue-wide counts, not spatially linked aftershocks. Follow-up is "
        "capped at the selected event's time; incomplete windows are censored, "
        "not treated as zero full-window activity."
    )
    rows = []
    for event_id in matches.event_id:
        followup = subsequent_activity(full_data, event_id, as_of=target.date_time)
        followup.insert(0, "similar_event_id", event_id)
        rows.append(followup)
    st.dataframe(pd.concat(rows, ignore_index=True), use_container_width=True, hide_index=True)


def modeling(full_data):
    st.caption(
        "Evaluation uses the full loaded catalogue and its fixed chronological "
        "holdout; sidebar filters do not change published model scores."
    )
    folder = ARTIFACTS / "model"
    if not (folder / "manifest.json").exists():
        st.info("Run the model command to generate evaluation results. Training is not triggered by dashboard visits.")
        st.code("python -m seismic model --data path/to/catalogue.csv")
        return
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != catalogue_fingerprint(
            full_data, tuple(manifest.get("extra_features", []))
        ):
            st.warning(
                "Saved results belong to a different catalogue. Run the model "
                "with the currently loaded CSV before comparing scores."
            )
            return
        metrics = pd.read_csv(folder / "metrics.csv")
        predictions = pd.read_csv(folder / "predictions.csv")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.warning(manifest["limitation"])
        value_columns = [c for c in predictions if c not in {"event_id", "date_time"}]
        chart(style(px.line(predictions, x="date_time", y=value_columns,
                            title="Held-out one-step magnitude evaluation",
                            labels={"value": "Magnitude", "date_time": "Recorded time"})))
        if manifest["epochs_run"]:
            history = pd.read_csv(folder / "history.csv")
            chart(style(px.line(history, x="epoch", y=["loss", "val_loss"],
                                title="Training and validation loss", labels={"value": "Scaled magnitude MSE"})))
        with st.expander("Protocol and reproducibility details"):
            st.json(manifest)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Saved model results are incomplete or invalid: {exc}. Rerun the model command.")


def quality_page(catalogue):
    quality = catalogue.quality
    st.caption(
        "Quality describes the full input catalogue before sidebar filters. "
        "Invalid rows and duplicate removal are reported explicitly."
    )
    st.json({key: value for key, value in quality.items() if key not in {"missing_values", "input_missing_values"}})
    st.subheader("Input missing values")
    st.dataframe(pd.Series(quality["input_missing_values"], name="missing_rows"))
    st.subheader("Normalized missing values")
    st.dataframe(pd.Series(quality["missing_values"], name="missing_rows"))


def render_page(page, data, catalogue):
    st.subheader(page)
    if page == "Data Quality":
        quality_page(catalogue)
    elif page == "Feature Workbench":
        from dashboard.research_pages import feature_workbench

        feature_workbench(catalogue.events)
    elif page == "Research Experiments":
        from dashboard.research_pages import research_experiments

        research_experiments(catalogue.events)
    elif page == "About":
        st.markdown("""A research portfolio for historical seismic catalogue analytics.
The bundled data contains Turkey-region events only; no Nepal events are supplied.
Load a genuine Nepal or global CSV with `date_time` (or `time`), `mag`, optional
`depth`, `latitude`, `longitude`, `place`, `id`, and `source` to enable those analyses.
Depth is in km and coordinates are WGS84 decimal degrees. Provide explicit timezone
offsets when known. Do not mix naive timestamps and timestamps with offsets.

The LSTM uses only lagged measured magnitude/depth, training-only scaling and
chronological validation/test blocks. It is compared with previous-value and
moving-average baselines. This application does not estimate structural loss,
calibrated hazard, or precise future earthquake occurrence.

[Earthquake prediction limitations — USGS](https://www.usgs.gov/faqs/can-you-predict-earthquakes)
""")
    elif page == "Forecasting / Model Evaluation":
        modeling(catalogue.events)
    elif data.empty:
        st.info("No events match the current filters. Broaden the filters to continue.")
    elif page == "Similarity Finder":
        similarity(data, catalogue.events)
    else:
        {"Overview": overview, "Earthquake Explorer": explorer, "Temporal Analysis": temporal,
         "Spatial Analysis": spatial, "Magnitude & Depth Analysis": magnitude_depth}[page](data)
