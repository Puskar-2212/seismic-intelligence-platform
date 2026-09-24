"""Command-line workflows sharing the dashboard's analysis implementation."""

import argparse
import json

import pandas as pd

from seismic import DISCLAIMER
from seismic.analysis import descriptive_statistics, event_sequences, strongest_events, temporal_summary
from seismic.data import load_catalogue
from seismic.paths import ARTIFACTS, DEFAULT_DATA, project_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Earthquake intelligence and exploratory analytics")
    parser.add_argument("command", choices=["preprocess", "analyze", "model", "fetch", "enrich", "experiment"])
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV path, relative to project root")
    parser.add_argument("--output", help="Output directory, relative to project root")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--period", choices=["hour", "day", "week"], default="day")
    parser.add_argument("--region", choices=["Global", "Nepal area (bounding box)", "Turkey area (bounding box)"],
                        default="Global")
    parser.add_argument("--waveforms", help="Measured waveform sample CSV")
    parser.add_argument("--geology", help="Event-linked geological feature CSV")
    parser.add_argument("--features", default="", help="Comma-separated additional feature names")
    parser.add_argument("--forecast-delay-seconds", type=float, default=0)
    parser.add_argument("--architecture", choices=["lstm", "gru", "cnn"], default="lstm")
    parser.add_argument("--units", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--architectures", default="lstm,gru,cnn")
    parser.add_argument("--search-units", default="8,16")
    parser.add_argument("--search-learning-rates", default="0.001")
    args = parser.parse_args(argv)
    try:
        output = project_path(args.output) if args.output else ARTIFACTS / args.command
        if args.command == "fetch":
            from seismic.live import fetch_feed, filter_region

            snapshot = fetch_feed(args.period)
            data = filter_region(snapshot.catalogue.events, args.region)
            output.mkdir(parents=True, exist_ok=True)
            data.to_csv(output / "events.csv", index=False)
            metadata = {"fetched_at": snapshot.fetched_at, "generated_at": snapshot.generated_at,
                        "source_url": snapshot.source_url, "region": args.region,
                        "filtered_records": len(data), "quality": snapshot.catalogue.quality}
            (output / "snapshot.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Downloaded {len(data)} observed events to {output}. Monitoring, not prediction.")
            return
        catalogue = load_catalogue(args.data)
        data = catalogue.events
        if data.empty:
            raise ValueError("No valid earthquake observations remain after validation.")
        from seismic.enrichment import attach_features, geology_features, waveform_features

        for path, extractor in ((args.waveforms, waveform_features), (args.geology, geology_features)):
            if path:
                raw = pd.read_csv(project_path(path), dtype={"event_id": "string"})
                data = attach_features(data, extractor(raw))
        extra_features = tuple(value.strip() for value in args.features.split(",") if value.strip())
        if args.command == "enrich" and not (args.waveforms or args.geology):
            raise ValueError("Provide --waveforms and/or --geology; no measurements are invented.")
        source = project_path(args.data).resolve()
        if source == (output / "events.csv").resolve():
            raise ValueError("Choose a different output directory; input files are not overwritten.")
        output.mkdir(parents=True, exist_ok=True)
        print(DISCLAIMER)
        if args.command == "model":
            from seismic.modeling import evaluate_models
            from seismic.networks import NetworkConfig

            metrics, _, manifest = evaluate_models(
                data, output, args.lookback, args.epochs, include_lstm=not args.baseline_only,
                config=NetworkConfig(args.architecture, args.units, args.learning_rate),
                extra_features=extra_features, forecast_delay_seconds=args.forecast_delay_seconds,
            )
            print(metrics.to_string(index=False))
            print(f"Train/validation/test windows: {manifest['train_windows']}/"
                  f"{manifest['validation_windows']}/{manifest['test_windows']}")
            print(manifest["limitation"])
        elif args.command == "experiment":
            from seismic.experiments import experiment_grid, run_experiments

            configurations = experiment_grid(
                tuple(args.architectures.split(",")),
                tuple(int(x) for x in args.search_units.split(",")),
                tuple(float(x) for x in args.search_learning_rates.split(",")),
            )
            leaderboard, metrics, _ = run_experiments(
                data, output, configurations, args.lookback, args.epochs,
                extra_features=extra_features, forecast_delay_seconds=args.forecast_delay_seconds,
            )
            print(leaderboard.to_string(index=False))
            print("Selected network and baselines on the final holdout:")
            print(metrics.to_string(index=False))
        else:
            data.to_csv(output / "events.csv", index=False)
            if args.command == "analyze":
                from seismic.visualization import eda_figures

                descriptive_statistics(data).to_csv(output / "statistics.csv")
                strongest_events(data).to_csv(output / "strongest_events.csv", index=False)
                temporal_summary(data).to_csv(output / "temporal_summary.csv", index=False)
                event_sequences(data).to_csv(output / "event_sequences.csv", index=False)
                for name, figure in eda_figures(data).items():
                    figure.write_html(output / f"{name}.html", include_plotlyjs=True)
            print(f"Retained {len(data)} of {catalogue.quality['input_records']} records.")
        (output / "quality.json").write_text(json.dumps(catalogue.quality, indent=2), encoding="utf-8")
        print(f"Outputs: {output}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
