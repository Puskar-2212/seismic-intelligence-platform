"""Near-real-time detected-event monitoring with explicit stale-data handling."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from seismic.live import REGIONS, fetch_feed, filter_region
from seismic.visualization import earthquake_map


@st.cache_data(ttl=60, show_spinner=False)
def get_snapshot(period):
    return fetch_feed(period)


def render_live():
    st.subheader("Live Monitoring")
    st.caption(
        "Official USGS detected events. Feeds may be delayed, incomplete or revised; "
        "this is not an earthquake prediction or early-warning service."
    )
    period = st.selectbox("Feed period", ["day", "hour", "week"])
    region = st.selectbox("Monitoring region", list(REGIONS))
    minimum = st.number_input("Minimum magnitude", value=0.0, step=0.5)
    connect = st.checkbox("Connect to USGS feed", value=False)
    automatic = st.checkbox("Refresh every 60 seconds while this page is open", value=True)
    if not connect:
        st.info("Enable the connection to load current observations from USGS.")
        return

    @st.fragment(run_every=60 if automatic else None)
    def observations():
        if st.button("Refresh now"):
            get_snapshot.clear(period)
        key = f"last_successful_usgs_{period}"
        try:
            snapshot = get_snapshot(period)
            st.session_state[key] = snapshot
        except ValueError as exc:
            st.error(str(exc))
            snapshot = st.session_state.get(key)
            if snapshot is None:
                st.info("No successful snapshot is available. Retry when the feed is reachable.")
                return
            st.warning("Connection failed. Showing the last successful snapshot, not current data.")
        now = pd.Timestamp(datetime.now(timezone.utc))
        age = (now - pd.Timestamp(snapshot.generated_at)).total_seconds()
        fetched_age = (now - pd.Timestamp(snapshot.fetched_at)).total_seconds()
        if age > 300 or fetched_age > 300:
            st.warning("This snapshot is over five minutes old. Treat it as stale.")
        elif age < -300:
            st.warning("Feed timestamp is in the future relative to this computer; check clock alignment.")
        st.caption(f"Retrieved: {snapshot.fetched_at} | Feed generated: {snapshot.generated_at}")
        st.markdown(f"[Source: USGS GeoJSON feed]({snapshot.source_url})")
        events = filter_region(snapshot.catalogue.events, region)
        events = events.loc[events.mag >= minimum]
        st.metric("Detected events matching filters", len(events))
        if region != "Global":
            st.caption("The regional bounding box includes neighbouring areas; it is not a country boundary.")
        if events.empty:
            st.info("No events match this feed window and filters; this does not prove absence of seismic activity.")
        else:
            figure = earthquake_map(events)
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
            columns = ["event_id", "date_time", "mag", "magnitude_type", "depth", "location",
                       "latitude", "longitude", "provider", "updated_at", "review_status"]
            st.dataframe(events[columns].sort_values("date_time", ascending=False),
                         use_container_width=True, hide_index=True)
            st.download_button("Download observed events", events.to_csv(index=False).encode("utf-8"),
                               "usgs_observed_events.csv", "text/csv")
        with st.expander("Feed quality and exclusions"):
            st.json(snapshot.catalogue.quality)

    observations()
