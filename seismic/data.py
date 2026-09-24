"""Conservative CSV normalization: never invent measurements or coordinates."""

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from seismic.paths import DEFAULT_DATA, project_path

ALIASES = {
    "date_time": ("timestamp", "time", "date"),
    "mag": ("magnitude",),
    "depth": ("depth_km",),
    "latitude": ("lat",),
    "longitude": ("lon", "lng"),
    "event_id": ("earthquake_id", "id"),
    "location": ("place", "title"),
    "provider": ("source", "net"),
}


@dataclass
class Catalogue:
    """Clean event table plus a report describing the original input."""

    events: pd.DataFrame
    quality: dict


def engineer_features(events: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar fields and descriptive bins without imputing values."""
    data = events.copy()
    for name in ("year", "month", "day", "hour", "day_of_week"):
        attr = "dayofweek" if name == "day_of_week" else name
        data[name] = getattr(data.date_time.dt, attr)
    data["magnitude_category"] = pd.cut(
        data.mag, [-np.inf, 2, 4, 5, 6, 7, np.inf], right=False,
        labels=["<2", "2–<4", "4–<5", "5–<6", "6–<7", "7+"],
    )
    data["depth_category"] = pd.Series(pd.NA, index=data.index, dtype="string")
    data.loc[data.depth < 70, "depth_category"] = "Shallow"
    data.loc[data.depth.between(70, 300), "depth_category"] = "Intermediate"
    data.loc[data.depth > 300, "depth_category"] = "Deep"
    return data


def normalize_catalogue(raw: pd.DataFrame) -> Catalogue:
    """Validate an input table and retain missing optional values explicitly.

    Naive dates remain recorded wall times. Aware dates normalize to UTC; mixing
    aware and naive values is rejected because the absolute ordering is unknown.
    Invalid timestamps/magnitudes are excluded. Invalid depths/coordinates become
    missing, with counts reported. Duplicate IDs retain their first valid row.
    """
    data = raw.copy().reset_index(drop=True)
    data.columns = data.columns.str.strip()
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    for target, aliases in ALIASES.items():
        if target not in data:
            for alias in aliases:
                if alias in data:
                    data[target] = data[alias]
                    break
    missing = {"date_time", "mag"} - set(data.columns)
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(sorted(missing))}")
    quality = {
        "input_records": len(raw), "available_columns": list(raw.columns),
        "input_missing_values": raw.isna().sum().astype(int).to_dict(),
        "exact_duplicate_rows": int(raw.duplicated().sum()), "problems": [],
    }
    parsed = data.date_time.map(lambda x: pd.to_datetime(x, errors="coerce"))
    aware = parsed.dropna().map(lambda x: x.tzinfo is not None)
    if aware.any() and not aware.all():
        raise ValueError("Mixed timezone-aware and naive timestamps; normalize the source first.")
    is_aware = bool(len(aware) and aware.all())
    data["date_time"] = pd.to_datetime(parsed, utc=is_aware, errors="coerce")
    quality["timestamp_basis"] = "UTC" if is_aware else "Recorded wall time; timezone unverified"
    if not is_aware:
        quality["problems"].append("Naive timestamps: timezone and absolute times are unverified.")
    for name in ("mag", "depth", "latitude", "longitude"):
        values = data[name] if name in data else pd.Series(np.nan, index=data.index)
        data[name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    invalid_depth = data.depth < 0
    invalid_coordinates = (
        (data.latitude.notna() & ~data.latitude.between(-90, 90))
        | (data.longitude.notna() & ~data.longitude.between(-180, 180))
    )
    quality["invalid_depth_rows"] = int(invalid_depth.sum())
    quality["invalid_coordinate_rows"] = int(invalid_coordinates.sum())
    data.loc[invalid_depth, "depth"] = np.nan
    data.loc[invalid_coordinates, ["latitude", "longitude"]] = np.nan
    invalid_required = data.date_time.isna() | data.mag.isna()
    quality["invalid_required_rows"] = int(invalid_required.sum())
    for name in ("location", "provider", "event_id"):
        if name not in data:
            data[name] = pd.NA
        data[name] = data[name].astype("string").str.strip().replace("", pd.NA)
    # Generated row keys are internal identifiers, never provider-issued IDs.
    data["id_generated"] = data.event_id.isna()
    supplied = set(data.event_id.dropna())
    for idx in data.index[data.id_generated]:
        key = f"local-row-{idx + 1}"
        while key in supplied:
            key = "local-" + key
        data.loc[idx, "event_id"] = key
        supplied.add(key)
    quality["duplicate_event_ids"] = int(
        data.loc[~data.id_generated, "event_id"].duplicated().sum()
    )
    data = data.loc[~invalid_required & ~raw.duplicated().to_numpy()].copy()
    data = data.drop_duplicates("event_id", keep="first")
    data = data.sort_values("date_time", kind="stable").reset_index(drop=True)
    data = engineer_features(data)
    quality.update({
        "clean_records": len(data), "excluded_rows": len(raw) - len(data),
        "date_start": str(data.date_time.min()) if len(data) else None,
        "date_end": str(data.date_time.max()) if len(data) else None,
        "missing_values": data.isna().sum().astype(int).to_dict(),
        "providers": sorted(data.provider.dropna().unique().tolist()),
        "missing_coordinate_percent": (
            float(data[["latitude", "longitude"]].isna().any(axis=1).mean() * 100)
            if len(data) else None
        ),
        "generated_ids": int(data.id_generated.sum()),
    })
    for field in ("invalid_required_rows", "invalid_depth_rows", "invalid_coordinate_rows",
                  "duplicate_event_ids", "exact_duplicate_rows"):
        if quality[field]:
            quality["problems"].append(f"{field}: {quality[field]}")
    if quality["missing_coordinate_percent"]:
        quality["problems"].append("Some or all events lack usable coordinates; no geocoding is applied.")
    if len(data) and data.date_time.max() - data.date_time.min() < pd.Timedelta(days=30):
        quality["problems"].append("Less than 30 days of observations; seasonality and follow-up are limited.")
    return Catalogue(data, quality)


def load_catalogue(source: str | Path | IO = DEFAULT_DATA) -> Catalogue:
    """Load a canonical, Kandilli-style or USGS-style CSV from disk or upload."""
    if isinstance(source, (str, Path)):
        source = project_path(source)
    try:
        raw = pd.read_csv(
            source, encoding="utf-8-sig",
            dtype={name: "string" for name in ("event_id", "earthquake_id", "id")},
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f"Could not read earthquake CSV: {exc}") from exc
    return normalize_catalogue(raw)
