"""Streamlit entry point: python -m streamlit run dashboard/app.py."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.pages import PAGES, render_page
from seismic import DISCLAIMER
from seismic.data import load_catalogue


def filter_events(events):
    """Apply sidebar filters while retaining unknown depths unless requested."""
    data = events.copy()
    dates = st.sidebar.date_input(
        "Date range", value=(data.date_time.min().date(), data.date_time.max().date()),
        min_value=data.date_time.min().date(), max_value=data.date_time.max().date(),
    )
    if len(dates) == 2:
        data = data.loc[data.date_time.dt.date.between(*dates)]
    for field, label in (("mag", "Magnitude range"), ("depth", "Depth range (km)")):
        values = events[field].dropna()
        if not values.empty and values.min() < values.max():
            limits = (float(values.min()), float(values.max()))
            selected = st.sidebar.slider(label, *limits, value=limits)
            mask = data[field].between(*selected)
            if field == "depth":
                keep_unknown = st.sidebar.checkbox("Include unknown depth", value=True)
                mask |= data[field].isna() & keep_unknown
            data = data.loc[mask]
    location = st.sidebar.text_input(
        "Location contains", help="Case-insensitive literal text; not a country classifier."
    )
    if location:
        data = data.loc[data.location.str.contains(location, case=False, regex=False, na=False)]
    st.sidebar.caption(f"{len(data)} of {len(events)} events match filters")
    return data


def main():
    st.set_page_config(page_title="Seismic Intelligence", page_icon="🌐", layout="wide")
    st.title("Earthquake Intelligence & Seismic Risk Analytics")
    st.caption("Historical catalogues · Measured patterns · Transparent evaluation")
    st.info(DISCLAIMER)
    st.sidebar.title("Seismic Intelligence")
    page = st.sidebar.radio("Navigate", PAGES)
    if page == "Live Monitoring":
        from dashboard.live_page import render_live

        render_live()
        return
    uploaded = st.sidebar.file_uploader("Load a Nepal / global / regional CSV", type="csv")
    st.sidebar.caption("Default: 70-event Turkey catalogue. Uploaded data stays in this session.")
    try:
        if uploaded is not None:
            uploaded.seek(0)
        catalogue = load_catalogue(uploaded) if uploaded is not None else load_catalogue()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if catalogue.events.empty:
        st.warning("No valid observations remain. Inspect the input quality report.")
        st.json(catalogue.quality)
        st.stop()
    filtered = filter_events(catalogue.events)
    st.caption(f"Timestamp basis: {catalogue.quality['timestamp_basis']}")
    if len(catalogue.events) < 500:
        st.warning(
            "Small catalogue: descriptive patterns and model scores are not "
            "evidence of reliable earthquake forecasting."
        )
    render_page(page, filtered, catalogue)


if __name__ == "__main__":
    main()
