"""Measured waveform summaries and event-linked geological covariates.

No waveform or geology values are supplied by default. Availability timestamps
are mandatory so experiments can exclude measurements released after issue time.
"""

import numpy as np
import pandas as pd

WAVEFORM_FEATURES = ("waveform_rms", "waveform_peak", "waveform_dominant_hz")
GEOLOGY_FEATURES = ("geo_fault_distance_km", "geo_vs30_m_s", "geo_elevation_m")


def _require(data: pd.DataFrame, columns) -> None:
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing enrichment columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("Enrichment input contains no records.")


def explicit_utc(values: pd.Series, name: str) -> pd.Series:
    """Require explicit offsets before normalizing measurement availability."""
    parsed = values.map(lambda value: pd.to_datetime(value, errors="coerce"))
    if parsed.isna().any() or not parsed.map(lambda value: value.tzinfo is not None).all():
        raise ValueError(f"{name} requires valid timestamps with explicit timezone offsets.")
    return pd.to_datetime(parsed, utc=True)


def _metadata(data: pd.DataFrame, columns) -> pd.DataFrame:
    result = data.copy()
    for column in columns:
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or result[column].eq("").any():
            raise ValueError(f"{column} cannot be missing or blank.")
    return result


def waveform_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize uniformly sampled traces; no instrument-response correction.

    Input CSV: event_id, station, channel, timestamp, amplitude, unit, source,
    available_at. One contiguous trace per event/station/channel. All traces
    must share units; per-event median summaries combine multiple traces.
    """
    _require(samples, ["event_id", "station", "channel", "timestamp", "amplitude",
                       "unit", "source", "available_at"])
    data = _metadata(samples, ["event_id", "station", "channel", "unit", "source"])
    data["timestamp"] = explicit_utc(data.timestamp, "timestamp")
    data["available_at"] = explicit_utc(data.available_at, "available_at")
    data["amplitude"] = pd.to_numeric(data.amplitude, errors="coerce")
    if not np.isfinite(data.amplitude).all():
        raise ValueError("Waveform amplitudes must be finite measured values.")
    if data.unit.nunique() != 1:
        raise ValueError("Convert traces to consistent physical units before combining them.")
    rows = []
    for (event_id, station, channel), trace in data.groupby(["event_id", "station", "channel"]):
        trace = trace.sort_values("timestamp")
        if len(trace) < 8:
            raise ValueError("Each waveform trace requires at least eight samples.")
        delta = np.diff(trace.timestamp.astype("int64").to_numpy()) / 1e9
        step = float(np.median(delta))
        if step <= 0 or not np.allclose(delta, step, rtol=0.01, atol=1e-6):
            raise ValueError("Waveform has duplicate times, gaps or nonuniform sampling; preprocess it first.")
        available = trace.available_at.max()
        if available < trace.timestamp.max():
            raise ValueError("Waveform availability cannot precede the last sample.")
        centered = trace.amplitude.to_numpy() - trace.amplitude.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        frequencies = np.fft.rfftfreq(len(centered), d=step)
        dominant = float(frequencies[1 + np.argmax(spectrum[1:])]) if np.any(centered) else 0.0
        rows.append({
            "event_id": event_id, "station": station, "channel": channel,
            "waveform_rms": float(np.sqrt(np.mean(centered ** 2))),
            "waveform_peak": float(np.max(np.abs(centered))),
            "waveform_dominant_hz": dominant, "waveform_available_at": available,
            "waveform_unit": trace.unit.iloc[0], "waveform_source": "; ".join(sorted(trace.source.unique())),
        })
    traces = pd.DataFrame(rows)
    aggregation = {feature: "median" for feature in WAVEFORM_FEATURES}
    aggregation.update({"waveform_available_at": "max", "waveform_unit": "first",
                        "waveform_source": lambda x: "; ".join(sorted(set(x)))})
    result = traces.groupby("event_id", as_index=False).agg(aggregation)
    result["waveform_trace_count"] = traces.groupby("event_id").size().reindex(result.event_id).to_numpy()
    return result


def geology_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate source-supplied per-event covariates; no inferred lithology/faults."""
    _require(raw, ["event_id", "source", "available_at"])
    data = _metadata(raw, ["event_id", "source"])
    if data.event_id.duplicated().any():
        raise ValueError("Geology input must have one row per event ID.")
    features = [column for column in GEOLOGY_FEATURES if column in data]
    if not features:
        raise ValueError(f"Provide at least one geological feature: {GEOLOGY_FEATURES}")
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"{column} requires finite measurements.")
        if column == "geo_fault_distance_km" and data[column].lt(0).any():
            raise ValueError("Fault distance cannot be negative.")
        if column == "geo_vs30_m_s" and data[column].le(0).any():
            raise ValueError("Vs30 must be positive.")
    data["geology_available_at"] = explicit_utc(data.available_at, "available_at")
    data["geology_source"] = data.source
    return data[["event_id", *features, "geology_available_at", "geology_source"]]


def attach_features(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One-to-one ID join; retain unmatched events as missing, report unknown IDs."""
    if features.event_id.duplicated().any() or events.event_id.duplicated().any():
        raise ValueError("Feature joins require unique event IDs.")
    unknown = set(features.event_id) - set(events.event_id)
    if unknown:
        raise ValueError(f"Enrichment contains {len(unknown)} event IDs absent from the catalogue.")
    overlaps = (set(features.columns) & set(events.columns)) - {"event_id"}
    if overlaps:
        raise ValueError(f"Feature columns already exist: {sorted(overlaps)}")
    return events.merge(features, on="event_id", how="left", validate="one_to_one", sort=False)
