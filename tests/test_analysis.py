import numpy as np
import pandas as pd
import pytest

from seismic.analysis import event_frequency, event_sequences, temporal_summary
from seismic.similarity import haversine_km, similar_events, subsequent_activity
from seismic.visualization import earthquake_map


def test_frequency_intervals_and_rolling_totals(events):
    assert event_frequency(events).events.sum() == len(events)
    sequences = event_sequences(events.iloc[::-1])
    assert sequences.hours_since_previous.iloc[1:].ge(0).all()
    summary = temporal_summary(events, 7)
    assert summary.rolling_events.iloc[-1] == 70
    assert summary.rolling_mean_magnitude.iloc[-1] == pytest.approx(events.mag.mean())


def test_similarity_uses_only_earlier_events(events):
    target = events.iloc[40]
    before = events.copy(deep=True)
    matches = similar_events(events, target.event_id)
    assert len(matches) == 5
    assert (matches.date_time < target.date_time).all()
    assert matches.distance.is_monotonic_increasing
    assert matches.similarity.between(0, 1).all()
    changed = events.copy()
    changed.loc[41:, ["mag", "depth"]] = 999
    pd.testing.assert_frame_equal(matches, similar_events(changed, target.event_id))
    pd.testing.assert_frame_equal(events, before)
    assert similar_events(events, events.iloc[0].event_id).empty


def test_known_similarity_distance_and_missing_depth(events):
    data = events.iloc[:4].copy()
    data["mag"] = [1, 2, 3, 2]
    data["depth"] = 10.0
    matches = similar_events(data, data.iloc[-1].event_id)
    assert matches.iloc[0].event_id == data.iloc[1].event_id
    assert matches.iloc[0].distance == 0
    assert matches.iloc[0].similarity == 1
    data.loc[data.index[-1], "depth"] = np.nan
    assert similar_events(data, data.iloc[-1].event_id).features_used.eq("mag").all()


def test_followup_censors_incomplete_windows_and_as_of(events):
    first_id = events.iloc[0].event_id
    followup = subsequent_activity(events, first_id)
    assert not followup.full_window_in_catalogue_span.any()
    assert followup.observed_events.tolist() == [69, 69, 69]
    cutoff = events.iloc[10].date_time
    assert subsequent_activity(events, first_id, cutoff).observed_events.tolist() == [10, 10, 10]
    assert subsequent_activity(events, events.iloc[-1].event_id).observed_events.eq(0).all()


def test_haversine_and_coordinate_handling(events):
    assert haversine_km(0, 0, np.array([0]), np.array([1]))[0] == pytest.approx(111.195, abs=0.01)
    assert haversine_km(0, 179, np.array([0]), np.array([-179]))[0] == pytest.approx(222.39, abs=0.01)
    assert earthquake_map(events) is None
    with pytest.raises(ValueError, match="no coordinates"):
        similar_events(events, events.iloc[-1].event_id, use_geography=True)
    # Coordinates here are a unit-test fixture, not an augmented real catalogue.
    fixture = events.iloc[:3].copy()
    fixture["latitude"] = [0, 0, 0]
    fixture["longitude"] = [0, 1, 0]
    assert earthquake_map(fixture) is not None
    matches = similar_events(fixture, fixture.iloc[-1].event_id, use_geography=True)
    assert "geographic_distance_km" in matches


def test_legacy_plot_does_not_mutate_input(events):
    import matplotlib.pyplot as plt
    from helpers.visualization_utils import plot_earthquake_frequency

    before = events.copy(deep=True)
    figure = plot_earthquake_frequency(events, show=False)
    pd.testing.assert_frame_equal(events, before)
    plt.close(figure)


def test_followup_full_window_and_endpoint_inclusion(events):
    # Test-only timestamp spacing checks exact boundaries and excludes simultaneous events.
    fixture = events.iloc[:5].copy()
    start = pd.Timestamp("2020-01-01")
    fixture["date_time"] = [start, start, start + pd.Timedelta(days=1),
                            start + pd.Timedelta(days=7), start + pd.Timedelta(days=30)]
    result = subsequent_activity(fixture, fixture.iloc[0].event_id)
    assert result.observed_events.tolist() == [1, 2, 3]
    assert result.full_window_in_catalogue_span.all()
