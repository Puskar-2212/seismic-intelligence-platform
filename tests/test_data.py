from io import StringIO

import numpy as np
import pandas as pd
import pytest

from seismic.data import engineer_features, load_catalogue, normalize_catalogue


def test_load_real_depths_and_chronological_order(raw):
    catalogue = load_catalogue()
    events = catalogue.events
    assert len(events) == 70
    assert events.date_time.is_monotonic_increasing
    original_depth = raw.set_index("earthquake_id").depth
    pd.testing.assert_series_equal(
        events.set_index("event_id").depth.sort_index(),
        original_depth.sort_index().rename_axis("event_id"), check_index_type=False,
    )
    assert catalogue.quality["missing_coordinate_percent"] == 100
    assert events.latitude.isna().all()
    assert catalogue.quality["providers"] == ["kandilli"]


def test_required_columns():
    with pytest.raises(ValueError, match="Required columns"):
        load_catalogue(StringIO("city,lat,lng\nKathmandu,27.7,85.3\n"))


def test_aliases_and_explicit_utc():
    # Tiny hand-written test fixture, never used for portfolio analysis/results.
    result = load_catalogue(StringIO(
        "id,time,magnitude,depth_km,latitude,longitude,place,net\n"
        "test,2020-01-01T01:00:00+01:00,2.5,70,27,85,Test location,test-provider\n"
    ))
    row = result.events.iloc[0]
    assert row.date_time == pd.Timestamp("2020-01-01T00:00:00Z")
    assert row.depth == 70 and row.latitude == 27 and row.longitude == 85
    assert row.depth_category == "Intermediate"
    assert row.provider == "test-provider"
    assert result.quality["timestamp_basis"] == "UTC"


def test_missing_depth_preserved(raw):
    events = normalize_catalogue(raw.drop(columns="depth")).events
    assert events.depth.isna().all()
    assert events.depth_category.isna().all()


def test_validation_and_duplicate_reporting(raw):
    raw.loc[0, "date_time"] = "invalid"
    raw.loc[1, "mag"] = np.inf
    raw.loc[2, "depth"] = -3
    raw["latitude"] = 28.0
    raw["longitude"] = 85.0
    raw.loc[3, "latitude"] = 200
    raw = pd.concat([raw, raw.iloc[[4]]], ignore_index=True)
    result = normalize_catalogue(raw)
    assert result.quality["invalid_required_rows"] == 2
    assert result.quality["invalid_depth_rows"] == 1
    assert result.quality["invalid_coordinate_rows"] == 1
    assert result.quality["exact_duplicate_rows"] == 1
    assert result.quality["duplicate_event_ids"] == 1
    assert len(result.events) == 68
    assert result.events.depth.isna().sum() == 1
    assert result.events.latitude.isna().sum() == 1


def test_calendar_and_category_boundaries(events):
    data = events.iloc[:5].copy()
    data["depth"] = [0, 69.9, 70, 300, 300.1]
    data["mag"] = [1.9, 2, 4, 6, 7]
    result = engineer_features(data)
    assert result.depth_category.tolist() == ["Shallow", "Shallow", "Intermediate", "Intermediate", "Deep"]
    assert result.magnitude_category.astype(str).tolist() == ["<2", "2–<4", "4–<5", "6–<7", "7+"]
    assert (result.year == result.date_time.dt.year).all()
    assert (result.day_of_week == result.date_time.dt.dayofweek).all()


def test_mixed_timezone_is_rejected(raw):
    raw.loc[0, "date_time"] = "2024-03-19T21:36:40Z"
    with pytest.raises(ValueError, match="Mixed timezone"):
        normalize_catalogue(raw)


def test_empty_and_all_invalid_inputs(raw):
    assert normalize_catalogue(raw.iloc[:0]).events.empty
    raw["mag"] = "unknown"
    result = normalize_catalogue(raw)
    assert result.events.empty
    assert result.quality["excluded_rows"] == 70


def test_generated_ids_do_not_collide(raw):
    raw = raw.drop(columns="earthquake_id")
    raw["event_id"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    raw.loc[1, "event_id"] = "local-row-1"
    result = normalize_catalogue(raw)
    assert len(result.events) == 70
    assert result.events.event_id.is_unique
    assert result.quality["generated_ids"] == 69


def test_provider_id_leading_zeros_are_preserved():
    result = load_catalogue(StringIO(
        "id,time,mag\n00012,2020-01-01T00:00:00Z,2.0\n"
    ))
    assert result.events.event_id.iloc[0] == "00012"
