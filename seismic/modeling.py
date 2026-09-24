"""Leakage-resistant one-step event-magnitude evaluation with simple baselines."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from seismic import DISCLAIMER
from seismic.enrichment import GEOLOGY_FEATURES, WAVEFORM_FEATURES, explicit_utc
from seismic.networks import NetworkConfig, config_dict, train_network


@dataclass
class SequenceSplit:
    events: pd.DataFrame
    scaler: MinMaxScaler
    x: np.ndarray
    y: np.ndarray
    target_indices: np.ndarray
    history_ends: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    train_end: int
    test_start: int


def catalogue_fingerprint(data: pd.DataFrame, extra_features: tuple = ()) -> str:
    """Bind results to the normalized input, including time and event identity."""
    columns = ["event_id", "date_time", "mag", "depth"]
    columns += list(extra_features)
    columns += sorted({"waveform_available_at" if c.startswith("waveform_")
                       else "geology_available_at" for c in extra_features})
    content = data[columns].to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chronological_boundaries(
    data: pd.DataFrame, test_fraction: float = 0.2, validation_fraction: float = 0.2,
) -> tuple[int, int]:
    """Last 20% test; last 20% of the preceding block validation by default.

    Entire tied-timestamp groups stay together. Fractions are approximate where
    timestamp ties require moving a boundary earlier.
    """
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if not data.date_time.is_monotonic_increasing or data.date_time.isna().any():
        raise ValueError("Chronological splitting requires sorted valid timestamps.")
    count = len(data)
    test_start = int(count * (1 - test_fraction))
    train_end = int(test_start * (1 - validation_fraction))
    boundaries = []
    for boundary in (train_end, test_start):
        while 0 < boundary < count and data.date_time.iloc[boundary - 1] == data.date_time.iloc[boundary]:
            boundary -= 1
        boundaries.append(boundary)
    if not 0 < boundaries[0] < boundaries[1] < count:
        raise ValueError("Too few distinct timestamps for train/validation/test blocks.")
    return tuple(boundaries)


def prepare_sequences(
    data: pd.DataFrame, lookback: int = 5, extra_features: tuple = (),
    forecast_delay_seconds: float = 0,
) -> SequenceSplit:
    """Use only strictly earlier observed magnitude/depth pairs as predictors.

    The scaler is fit only on the training block, excluding validation and test.
    Later held-out targets can use previously observed test events: this is
    rolling one-step evaluation, not a fixed-origin multi-step forecast.
    """
    if lookback < 1 or not np.isfinite(forecast_delay_seconds) or forecast_delay_seconds < 0:
        raise ValueError("lookback must be positive and forecast delay finite and nonnegative.")
    allowed = set(WAVEFORM_FEATURES + GEOLOGY_FEATURES)
    if set(extra_features) - allowed or len(set(extra_features)) != len(extra_features):
        raise ValueError("Extra features must be unique supported waveform/geology feature names.")
    columns = ["mag", "depth", *extra_features]
    availability_columns = sorted({"waveform_available_at" if c.startswith("waveform_")
                                   else "geology_available_at" for c in extra_features})
    missing = set(columns + availability_columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing model feature columns: {sorted(missing)}")
    events = (data.dropna(subset=["date_time", "mag", "depth"])
              .sort_values("date_time", kind="stable").reset_index(drop=True))
    if not np.isfinite(events[["mag", "depth"]].to_numpy()).all():
        raise ValueError("Model inputs must be finite.")
    train_end, test_start = chronological_boundaries(events)
    if train_end < lookback + 5:
        raise ValueError("Too few complete observations: need at least five training windows plus validation/test.")
    for column in extra_features:
        events[column] = pd.to_numeric(events[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if extra_features and events.date_time.dt.tz is None:
        raise ValueError("Enriched modeling requires verified timezone-aware event timestamps.")
    for column in availability_columns:
        present = events[column].notna()
        parsed = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
        if present.any():
            parsed.loc[present] = explicit_utc(events.loc[present, column], column)
        events[column] = parsed
    history_ends = np.asarray(events.date_time.searchsorted(events.date_time, side="left"))
    targets = np.flatnonzero(history_ends >= lookback)
    eligible = []
    for target in targets:
        end = history_ends[target]
        window = events.iloc[end - lookback:end]
        issue_time = window.date_time.iloc[-1] + pd.Timedelta(seconds=forecast_delay_seconds)
        complete = np.isfinite(window[columns].to_numpy(dtype=float)).all()
        available = all(window[c].notna().all() and window[c].le(issue_time).all()
                        for c in availability_columns)
        if complete and available and issue_time < events.date_time.iloc[target]:
            eligible.append(target)
    targets = np.asarray(eligible, dtype=int)
    ends = history_ends[targets]
    masks = (targets < train_end, (targets >= train_end) & (targets < test_start), targets >= test_start)
    if any(not mask.any() for mask in masks):
        raise ValueError(
            "Every chronological split needs eligible windows; "
            "check missing features and availability times."
        )
    if extra_features:
        # Only values actually available in training windows may influence scaling.
        fitting_rows = sorted({row for end in ends[masks[0]] for row in range(end - lookback, end)})
        fitting_data = events.loc[fitting_rows, columns]
    else:
        fitting_data = events.loc[:train_end - 1, columns]
    scaler = MinMaxScaler().fit(fitting_data)
    scaled = scaler.transform(events[columns])
    x = np.stack([scaled[end - lookback:end] for end in ends]).astype("float32")
    y = scaled[targets, 0].astype("float32")
    return SequenceSplit(events, scaler, x, y, targets, ends, *masks, train_end, test_start)


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Report errors in original magnitude units (MSE in squared units)."""
    mse = float(mean_squared_error(actual, predicted))
    return {"MAE": float(mean_absolute_error(actual, predicted)), "MSE": mse, "RMSE": float(np.sqrt(mse))}


def evaluate_models(
    data: pd.DataFrame, output_dir: Path, lookback: int = 5,
    epochs: int = 100, seed: int = 42, include_lstm: bool = True,
    config: NetworkConfig = NetworkConfig(), extra_features: tuple = (),
    forecast_delay_seconds: float = 0, _training_result=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate the selected sequence architecture and baselines on identical targets.

    include_lstm retains the original API name; it enables the configured network.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    split = prepare_sequences(data, lookback, extra_features, forecast_delay_seconds)
    targets = split.target_indices[split.test_mask]
    ends = split.history_ends[split.test_mask]
    magnitudes = split.events.mag.to_numpy()
    predictions = split.events.loc[targets, ["event_id", "date_time", "mag"]].reset_index(drop=True)
    predictions = predictions.rename(columns={"mag": "actual"})
    predictions["previous_value"] = magnitudes[ends - 1]
    predictions["moving_average"] = [magnitudes[end - lookback:end].mean() for end in ends]
    history = pd.DataFrame()
    model = None
    if include_lstm:
        trained = _training_result or train_network(split, config, epochs, seed)
        model, history = trained.model, trained.history
        scaled_prediction = model(split.x[split.test_mask], training=False).numpy().ravel()
        predictions[config.architecture] = (scaled_prediction - split.scaler.min_[0]) / split.scaler.scale_[0]
    model_names = [c for c in predictions if c not in {"event_id", "date_time", "actual"}]
    metrics = pd.DataFrame([
        {"model": name, **regression_metrics(predictions.actual, predictions[name])}
        for name in model_names
    ])
    manifest = {
        "disclaimer": DISCLAIMER,
        "limitation": (
            "Small or incomplete catalogues cannot establish reliable forecast skill. "
            "No event times or locations are predicted."
        ),
        "protocol": (
            "Chronological rolling one-step event magnitude, strictly earlier "
            "observations only; no recursive forecasting."
        ),
        "dataset_sha256": catalogue_fingerprint(data, extra_features),
        "extra_features": list(extra_features),
        "forecast_delay_seconds": forecast_delay_seconds,
        "network_config": config_dict(config),
        "input_records": len(data), "complete_model_records": len(split.events),
        "excluded_missing_measurements": len(data) - len(split.events),
        "lookback_events": lookback, "seed": seed, "max_epochs": epochs,
        "epochs_run": len(history), "parameters": model.count_params() if model else None,
        "train_records": split.train_end,
        "validation_records": split.test_start - split.train_end,
        "test_records": len(split.events) - split.test_start,
        "train_windows": int(split.train_mask.sum()),
        "validation_windows": int(split.validation_mask.sum()),
        "test_windows": int(split.test_mask.sum()),
        "test_eligible_fraction": float(split.test_mask.sum() / (len(split.events) - split.test_start)),
        "evaluated_test_start": str(predictions.date_time.iloc[0]),
        "evaluated_test_end": str(predictions.date_time.iloc[-1]),
        "eligibility_policy": "Complete lag features available by issue time; target must occur after issue time.",
        "train_end": str(split.events.date_time.iloc[split.train_end - 1]),
        "validation_start": str(split.events.date_time.iloc[split.train_end]),
        "test_start": str(split.events.date_time.iloc[split.test_start]),
        "test_end": str(split.events.date_time.iloc[-1]),
        "feature_order": ["mag", "depth", *extra_features],
        "scaler_min": split.scaler.min_.tolist(), "scaler_scale": split.scaler.scale_.tolist(),
        "scaler_training_min": split.scaler.data_min_.tolist(),
        "scaler_training_max": split.scaler.data_max_.tolist(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scikit-learn"] + (["tensorflow", "keras"] if include_lstm else [])},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The manifest is written last: incomplete runs are not valid dashboard results.
    (output_dir / "manifest.json").unlink(missing_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    history.to_csv(output_dir / "history.csv")
    _save_model_plots(predictions, history, model_names, output_dir)
    if model is not None:
        model.save(output_dir / "lstm.keras")  # Legacy artifact name; inspect network_config for architecture.
    else:
        (output_dir / "lstm.keras").unlink(missing_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics, predictions, manifest


def _save_model_plots(predictions, history, model_names, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(predictions.date_time, predictions.actual, label="Observed", marker="o", color="black")
    for name in model_names:
        ax.plot(predictions.date_time, predictions[name], label=name.replace("_", " "))
    ax.set(title="Held-out one-step magnitude evaluation", xlabel="Recorded time", ylabel="Magnitude")
    ax.legend()
    fig.autofmt_xdate()
    fig.text(0.5, 0.01, "Exploratory evaluation; cannot predict exact future earthquakes.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    if not history.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history.index, history.loss, label="Training")
        ax.plot(history.index, history.val_loss, label="Validation")
        ax.set(title="Sequence model learning curves", xlabel="Epoch", ylabel="Scaled magnitude MSE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss.png", dpi=150)
        plt.close(fig)
    else:
        (output_dir / "training_loss.png").unlink(missing_ok=True)
