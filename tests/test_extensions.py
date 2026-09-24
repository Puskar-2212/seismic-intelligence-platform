"""Explicit test fixtures, never portfolio earthquake/waveform observations."""

import copy
from datetime import datetime, timezone
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from seismic.enrichment import attach_features, geology_features, waveform_features
from seismic.live import FEEDS, fetch_feed, filter_region, parse_feed
from seismic.modeling import catalogue_fingerprint, prepare_sequences
from seismic.paths import PROJECT_ROOT


@pytest.fixture
def feed_payload():
    return {
        "type": "FeatureCollection", "metadata": {"generated": 1700000000000},
        "features": [{
            "id": "test-only-id", "geometry": {"type": "Point", "coordinates": [85, 28, 12]},
            "properties": {"type": "earthquake", "time": 1700000000000, "updated": 1700000001000,
                           "mag": 3, "place": "Test fixture", "net": "test", "status": "reviewed"},
        }],
    }


@pytest.fixture
def samples():
    start = pd.Timestamp("2020-01-01T00:00:00Z")
    return pd.DataFrame({
        "event_id": "test-only-id", "station": "TEST", "channel": "BHZ",
        "timestamp": pd.date_range(start, periods=64, freq=pd.Timedelta(seconds=1 / 64)),
        "amplitude": 2 * np.sin(2 * np.pi * 8 * np.arange(64) / 64),
        "unit": "m/s", "source": "mathematical unit-test signal",
        "available_at": start + pd.Timedelta(seconds=2),
    })


def test_feed_coordinates_utc_and_latest_revision(feed_payload):
    older = copy.deepcopy(feed_payload["features"][0])
    older["properties"].update(updated=1690000000000, mag=2)
    feed_payload["features"].append(older)
    snapshot = parse_feed(feed_payload, FEEDS["day"], "2023-11-14T22:14:00Z")
    events = snapshot.catalogue.events
    assert len(events) == 1 and events.iloc[0].mag == 3
    assert events.iloc[0].latitude == 28 and events.iloc[0].longitude == 85
    assert str(events.date_time.dt.tz) == "UTC"
    assert len(filter_region(events, "Nepal area (bounding box)")) == 1
    assert filter_region(events, "Turkey area (bounding box)").empty
    assert snapshot.catalogue.quality["duplicate_event_ids"] == 1


def test_empty_feed_and_malformed_features(feed_payload):
    feed_payload["features"] = [{"invalid": "fixture"}]
    snapshot = parse_feed(feed_payload, FEEDS["day"], "2023-11-14T22:14:00Z")
    assert snapshot.catalogue.events.empty
    assert snapshot.catalogue.quality["malformed_feed_features"] == 1
    with pytest.raises(ValueError, match="FeatureCollection"):
        parse_feed([], "", "")


def test_live_network_failure_has_no_fake_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("test offline")

    monkeypatch.setattr("seismic.live.urlopen", fail)
    with pytest.raises(ValueError, match="unavailable"):
        fetch_feed()
    with pytest.raises(ValueError, match="period"):
        fetch_feed("untrusted-url")


def test_waveform_known_signal_features(samples):
    original = samples.copy(deep=True)
    features = waveform_features(samples)
    assert features.waveform_rms.iloc[0] == pytest.approx(np.sqrt(2))
    assert features.waveform_peak.iloc[0] == pytest.approx(2)
    assert features.waveform_dominant_hz.iloc[0] == pytest.approx(8)
    assert features.waveform_trace_count.iloc[0] == 1
    pd.testing.assert_frame_equal(samples, original)


@pytest.mark.parametrize("problem", ["gap", "duplicate", "units", "release", "missing_source", "naive"])
def test_waveform_rejects_invalid_input(samples, problem):
    if problem == "gap":
        samples.loc[5, "timestamp"] += pd.Timedelta(milliseconds=3)
    elif problem == "duplicate":
        samples.loc[5, "timestamp"] = samples.loc[4, "timestamp"]
    elif problem == "units":
        samples.loc[5, "unit"] = "counts"
    elif problem == "release":
        samples["available_at"] = samples.timestamp.min()
    elif problem == "missing_source":
        samples["source"] = ""
    else:
        samples["timestamp"] = samples.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError):
        waveform_features(samples)


def test_geology_validation_and_id_join(events):
    fixture = pd.DataFrame({"event_id": [events.event_id.iloc[0]], "source": ["test fixture"],
                            "available_at": ["2020-01-01T00:00:00Z"], "geo_vs30_m_s": [500]})
    joined = attach_features(events, geology_features(fixture))
    assert len(joined) == 70
    assert joined.geo_vs30_m_s.notna().sum() == 1
    fixture["geo_vs30_m_s"] = -1
    with pytest.raises(ValueError, match="positive"):
        geology_features(fixture)
    fixture["geo_vs30_m_s"] = 500
    fixture["event_id"] = "unknown-test-id"
    with pytest.raises(ValueError, match="absent"):
        attach_features(events, geology_features(fixture))


def test_extra_feature_availability_and_training_only_scaling(events):
    # Test-only timezone assignment, not a correction to the actual catalogue.
    data = events.copy()
    data["date_time"] = data.date_time.dt.tz_localize("UTC")
    data["waveform_rms"] = np.arange(70, dtype=float)
    data["waveform_available_at"] = data.date_time + pd.Timedelta(seconds=10)
    with pytest.raises(ValueError, match="availability"):
        prepare_sequences(data, extra_features=("waveform_rms",))
    split = prepare_sequences(data, extra_features=("waveform_rms",), forecast_delay_seconds=20)
    assert split.x.shape[-1] == 3
    for target, end in zip(split.target_indices, split.history_ends):
        issued = data.date_time.iloc[end - 1] + pd.Timedelta(seconds=20)
        assert data.date_time.iloc[target] > issued
        assert data.waveform_available_at.iloc[end - 5:end].le(issued).all()
    changed = data.copy()
    changed.loc[44:, "waveform_rms"] = 9999
    other = prepare_sequences(changed, extra_features=("waveform_rms",), forecast_delay_seconds=20)
    np.testing.assert_array_equal(split.scaler.data_max_, other.scaler.data_max_)
    np.testing.assert_array_equal(split.x[split.train_mask], other.x[other.train_mask])
    assert catalogue_fingerprint(data, ("waveform_rms",)) != catalogue_fingerprint(changed, ("waveform_rms",))


def test_enriched_model_rejects_unverified_timezone(events):
    events["geo_vs30_m_s"] = 500
    events["geology_available_at"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_sequences(events, extra_features=("geo_vs30_m_s",))


@pytest.mark.model
def test_all_architectures_and_validation_selection(events, tmp_path):
    from seismic.experiments import experiment_grid, run_experiments

    configurations = experiment_grid(units=(8,))
    comparison, metrics, selection = run_experiments(events, tmp_path, configurations, epochs=2)
    assert set(comparison.architecture) == {"lstm", "gru", "cnn"}
    assert comparison.iloc[0].architecture == selection["selected_config"]["architecture"]
    assert not selection["test_used_for_selection"]
    assert len(metrics) == 3  # Selected network plus two baselines, not all candidates.
    assert not any(column.startswith("test_") for column in comparison)
    assert (tmp_path / "selected" / "lstm.keras").exists()


def test_live_page_handles_stale_then_offline_snapshot(feed_payload, monkeypatch):
    import dashboard.live_page as live_page

    snapshot = parse_feed(feed_payload, FEEDS["day"], datetime.now(timezone.utc).isoformat())
    monkeypatch.setattr(live_page, "get_snapshot", lambda period: snapshot)
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Live Monitoring").run()
    app.checkbox[0].set_value(True).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert any("stale" in warning.value for warning in app.warning)

    def fail(period):
        raise ValueError("test offline")

    monkeypatch.setattr(live_page, "get_snapshot", fail)
    app.run()
    assert not app.exception
    assert any("last successful snapshot" in warning.value for warning in app.warning)
    assert app.metric[0].value == "1"
