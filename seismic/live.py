"""Bounded, read-only access to official USGS detected-earthquake feeds."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from seismic.data import Catalogue, normalize_catalogue

FEEDS = {
    period: f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_{period}.geojson"
    for period in ("hour", "day", "week")
}
REGIONS = {"Global": None, "Nepal area (bounding box)": (26, 31, 80, 89),
           "Turkey area (bounding box)": (35, 43, 25, 45)}


@dataclass
class FeedSnapshot:
    catalogue: Catalogue
    fetched_at: str
    generated_at: str
    source_url: str


def parse_feed(payload: dict, source_url: str, fetched_at: str) -> FeedSnapshot:
    """Normalize GeoJSON; preserve event IDs, UTC times and source revisions."""
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("USGS response is not a GeoJSON FeatureCollection.")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("USGS response has no feature list.")
    rows, malformed = [], 0
    for feature in features:
        try:
            props = feature["properties"]
            if props.get("type") != "earthquake":
                continue
            geometry = feature["geometry"]
            coordinates = geometry["coordinates"]
            if geometry["type"] != "Point" or len(coordinates) < 3 or not feature.get("id"):
                raise ValueError("Missing point geometry or ID")
            rows.append({
                "event_id": str(feature["id"]),
                "date_time": pd.to_datetime(props["time"], unit="ms", utc=True, errors="coerce"),
                "updated_at": pd.to_datetime(props.get("updated"), unit="ms", utc=True, errors="coerce"),
                "mag": props.get("mag"), "depth": coordinates[2],
                "longitude": coordinates[0], "latitude": coordinates[1],
                "location": props.get("place"), "provider": props.get("net") or "USGS",
                "magnitude_type": props.get("magType"), "review_status": props.get("status"),
            })
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            malformed += 1
    columns = ["event_id", "date_time", "updated_at", "mag", "depth", "longitude",
               "latitude", "location", "provider", "magnitude_type", "review_status"]
    raw = pd.DataFrame(rows, columns=columns)
    # Latest revision wins before the general loader's duplicate-ID policy.
    raw = raw.sort_values("updated_at", ascending=False, na_position="last")
    catalogue = normalize_catalogue(raw)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("USGS response is missing feed metadata.")
    generated = pd.to_datetime(metadata.get("generated"), unit="ms", utc=True, errors="coerce")
    if pd.isna(generated):
        raise ValueError("USGS response is missing its generation timestamp.")
    catalogue.quality["malformed_feed_features"] = malformed
    catalogue.quality["feed_feature_count"] = len(features)
    catalogue.quality["timestamp_basis"] = "UTC"
    if not rows:
        catalogue.quality["problems"] = [
            problem for problem in catalogue.quality["problems"] if "Naive timestamps" not in problem
        ]
    return FeedSnapshot(catalogue, fetched_at, generated.isoformat(), source_url)


def fetch_feed(period: str = "day", timeout: float = 15) -> FeedSnapshot:
    """Fetch at most 20 MB with a timeout. No implicit retry or fake fallback."""
    if period not in FEEDS:
        raise ValueError("Feed period must be hour, day or week.")
    request = Request(FEEDS[period], headers={"User-Agent": "SeismicIntelligencePortfolio/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000:
            raise ValueError("USGS response exceeds the 20 MB limit.")
        payload = json.loads(content)
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"USGS feed unavailable: {exc}") from exc
    return parse_feed(payload, FEEDS[period], datetime.now(timezone.utc).isoformat())


def filter_region(events: pd.DataFrame, region: str) -> pd.DataFrame:
    """Bounding-box selection, not an authoritative country boundary."""
    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")
    bounds = REGIONS[region]
    if bounds is None:
        return events.copy()
    south, north, west, east = bounds
    return events.loc[events.latitude.between(south, north)
                      & events.longitude.between(west, east)].copy()
