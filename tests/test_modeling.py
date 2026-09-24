import json

import numpy as np
import pytest

from seismic.modeling import (
    catalogue_fingerprint, chronological_boundaries, evaluate_models,
    prepare_sequences, regression_metrics,
)


def test_chronological_split_and_training_only_scaler(events):
    split = prepare_sequences(events)
    assert (split.train_end, split.test_start) == (44, 56)
    assert (split.train_mask.sum(), split.validation_mask.sum(), split.test_mask.sum()) == (39, 12, 14)
    altered = events.copy()
    altered.loc[44:, "mag"] = 20
    other = prepare_sequences(altered)
    np.testing.assert_array_equal(split.scaler.data_max_, other.scaler.data_max_)
    np.testing.assert_array_equal(split.x[split.train_mask], other.x[other.train_mask])
    assert other.y[other.test_mask].min() > 1  # no clipping/refitting to test range


def test_windows_never_use_target_or_future_measurements(events):
    original = prepare_sequences(events)
    altered = events.copy()
    altered.loc[56:, "depth"] = 999
    changed = prepare_sequences(altered)
    np.testing.assert_array_equal(original.x[original.test_mask][0], changed.x[changed.test_mask][0])
    for target, end in zip(original.target_indices, original.history_ends):
        assert events.date_time.iloc[end - 1] < events.date_time.iloc[target]


def test_timestamps_ties_stay_together_and_out_of_histories(events):
    tied = events.copy()
    tied.loc[55:57, "date_time"] = tied.loc[55, "date_time"]
    split = prepare_sequences(tied)
    assert split.test_start == 55
    for target, end in zip(split.target_indices, split.history_ends):
        assert tied.date_time.iloc[end - 1] < tied.date_time.iloc[target]


def test_reject_unsorted_and_too_small_data(events):
    with pytest.raises(ValueError, match="sorted"):
        chronological_boundaries(events.iloc[::-1])
    with pytest.raises(ValueError, match="Too few"):
        prepare_sequences(events.head(10))


def test_metrics_original_units():
    assert regression_metrics([1, 3], [2, 1]) == {"MAE": 1.5, "MSE": 2.5, "RMSE": np.sqrt(2.5)}


def test_baselines_share_exact_test_targets(events, tmp_path):
    metrics, predictions, manifest = evaluate_models(events, tmp_path, include_lstm=False)
    assert predictions.event_id.tolist() == events.event_id.iloc[56:].tolist()
    np.testing.assert_allclose(predictions.previous_value, events.mag.iloc[55:-1])
    assert predictions.moving_average.iloc[0] == pytest.approx(events.mag.iloc[51:56].mean())
    assert metrics.model.tolist() == ["previous_value", "moving_average"]
    assert manifest["dataset_sha256"] == catalogue_fingerprint(events)
    assert (tmp_path / "actual_vs_predicted.png").exists()


@pytest.mark.model
def test_real_lstm_training_and_saved_outputs(events, tmp_path):
    metrics, predictions, manifest = evaluate_models(events, tmp_path, epochs=2)
    assert manifest["epochs_run"] == 2
    assert manifest["parameters"] == 361
    assert set(metrics.model) == {"previous_value", "moving_average", "lstm"}
    assert np.isfinite(predictions.lstm).all()
    assert (tmp_path / "lstm.keras").exists()
    assert (tmp_path / "training_loss.png").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["test_windows"] == 14
    # Persisted model plus recorded scaler must reproduce the published outputs.
    import tensorflow as tf

    restored = tf.keras.models.load_model(tmp_path / "lstm.keras")
    split = prepare_sequences(events)
    scaled = restored(split.x[split.test_mask], training=False).numpy().ravel()
    actual = (scaled - split.scaler.min_[0]) / split.scaler.scale_[0]
    np.testing.assert_allclose(predictions.lstm, actual, atol=1e-6)
