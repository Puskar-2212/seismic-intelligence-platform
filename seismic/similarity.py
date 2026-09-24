"""Nearest historical events, with explicit scaling and follow-up coverage."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def haversine_km(lat: float, lon: float, latitudes, longitudes) -> np.ndarray:
    """Great-circle distance on a spherical Earth, including the date line."""
    phi, lam = np.radians(lat), np.radians(lon)
    other_phi, other_lam = np.radians(latitudes), np.radians(longitudes)
    a = (np.sin((other_phi - phi) / 2) ** 2
         + np.cos(phi) * np.cos(other_phi) * np.sin((other_lam - lam) / 2) ** 2)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def similar_events(
    data: pd.DataFrame, event_id: str, count: int = 5,
    use_geography: bool = False, geographic_scale_km: float = 100,
) -> pd.DataFrame:
    """Rank strictly earlier events by standardized magnitude/depth distance.

    Scaling uses earlier candidates only. Missing target depth omits depth for
    all candidates. Geography requires coordinates for target and candidates;
    the additional term is distance / geographic_scale_km. Score 1/(1+d) is a
    descriptive similarity, never a probability.
    """
    if count < 1 or geographic_scale_km <= 0:
        raise ValueError("count and geographic_scale_km must be positive.")
    selected = data.loc[data.event_id == event_id]
    if len(selected) != 1:
        raise ValueError("Select one unique event ID.")
    target = selected.iloc[0]
    features = ["mag"] + (["depth"] if pd.notna(target.depth) else [])
    candidates = data.loc[data.date_time < target.date_time].dropna(subset=features).copy()
    if use_geography:
        if pd.isna(target.latitude) or pd.isna(target.longitude):
            raise ValueError("Selected event has no coordinates; disable geographic matching.")
        candidates = candidates.dropna(subset=["latitude", "longitude"])
    if candidates.empty:
        return candidates.assign(distance=pd.Series(dtype=float), similarity=pd.Series(dtype=float))
    scaler = StandardScaler().fit(candidates[features])
    difference = scaler.transform(candidates[features]) - scaler.transform(selected[features])
    squared = (difference ** 2).sum(axis=1)
    if use_geography:
        candidates["geographic_distance_km"] = haversine_km(
            target.latitude, target.longitude, candidates.latitude, candidates.longitude
        )
        squared += (candidates.geographic_distance_km / geographic_scale_km) ** 2
    candidates["distance"] = np.sqrt(squared)
    candidates["similarity"] = 1 / (1 + candidates.distance)
    candidates["features_used"] = ", ".join(features) + (", geography" if use_geography else "")
    return candidates.sort_values(["distance", "date_time"], kind="stable").head(count)


def subsequent_activity(
    data: pd.DataFrame, event_id: str, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Observed catalogue-wide activity in (event time, event time + days].

    Counts are censored at the catalogue end and optional as_of cutoff. Full
    span coverage does not establish catalogue completeness or physical linkage.
    """
    selected = data.loc[data.event_id == event_id]
    if len(selected) != 1:
        raise ValueError("Select one unique event ID.")
    start = selected.iloc[0].date_time
    end = data.date_time.max() if as_of is None else min(data.date_time.max(), as_of)
    if end < start:
        raise ValueError("Follow-up cutoff precedes the selected event.")
    rows = []
    for days in (1, 7, 30):
        horizon = start + pd.Timedelta(days=days)
        observed_end = min(horizon, end)
        following = data.loc[(data.date_time > start) & (data.date_time <= observed_end)]
        rows.append({
            "window_days": days, "observed_events": len(following),
            "max_magnitude": following.mag.max(),
            "observed_hours": (observed_end - start).total_seconds() / 3600,
            "full_window_in_catalogue_span": bool(end >= horizon),
        })
    return pd.DataFrame(rows)
