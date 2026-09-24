"""Validation-only architecture selection with one final holdout evaluation."""

from itertools import product
import json
from pathlib import Path

import pandas as pd

from seismic.modeling import evaluate_models, prepare_sequences, regression_metrics
from seismic.networks import NetworkConfig, config_dict, train_network


def experiment_grid(architectures=("lstm", "gru", "cnn"), units=(8, 16), rates=(0.001,)):
    configurations = [NetworkConfig(name, size, rate)
                      for name, size, rate in product(architectures, units, rates)]
    if not configurations or len(configurations) > 24 or len(set(configurations)) != len(configurations):
        raise ValueError("Use 1–24 unique experiment configurations.")
    for config in configurations:
        config.validate()
    return configurations


def run_experiments(
    data, output_dir: Path, configurations=None, lookback=5, epochs=100, seed=42,
    extra_features=(), forecast_delay_seconds=0,
):
    """Rank by validation RMSE; no test scores are produced for losing candidates.

    Repeated use of the final holdout for human-guided tuning invalidates its
    independence. The bundled holdout has already been viewed in prior work.
    """
    configurations = experiment_grid() if configurations is None else list(configurations)
    if not 1 <= len(configurations) <= 24 or len(set(configurations)) != len(configurations):
        raise ValueError("Use 1–24 unique experiment configurations.")
    for config in configurations:
        config.validate()
    split = prepare_sequences(data, lookback, extra_features, forecast_delay_seconds)
    actual = split.events.mag.iloc[split.target_indices[split.validation_mask]].to_numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").unlink(missing_ok=True)
    rows, best, best_rmse = [], None, float("inf")
    for index, config in enumerate(configurations):
        print(f"Candidate {index + 1}/{len(configurations)}: {config}", flush=True)
        trained = train_network(split, config, epochs, seed)
        scaled = trained.model(split.x[split.validation_mask], training=False).numpy().ravel()
        predictions = (scaled - split.scaler.min_[0]) / split.scaler.scale_[0]
        scores = regression_metrics(actual, predictions)
        rows.append({"candidate": index, **config_dict(config), "epochs_run": len(trained.history),
                     **{f"validation_{key}": value for key, value in scores.items()}})
        trained.history.to_csv(output_dir / f"candidate_{index}_history.csv")
        if scores["RMSE"] < best_rmse:
            best, best_rmse = trained, scores["RMSE"]
    leaderboard = pd.DataFrame(rows).sort_values("validation_RMSE", kind="stable")
    leaderboard.to_csv(output_dir / "validation_comparison.csv", index=False)
    metrics, predictions, manifest = evaluate_models(
        data, output_dir / "selected", lookback, epochs, seed,
        config=best.config, extra_features=extra_features,
        forecast_delay_seconds=forecast_delay_seconds, _training_result=best,
    )
    selection = {"selected_config": config_dict(best.config),
                 "selection_metric": "validation_RMSE", "candidate_count": len(rows),
                 "seed": seed, "test_used_for_selection": False,
                 "dataset_sha256": manifest["dataset_sha256"],
                 "extra_features": list(extra_features),
                 "limitation": (
                     "The previously inspected 70-event holdout is demonstration data, "
                     "not fresh confirmation."
                 )}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return leaderboard, metrics, selection
