"""Compatibility plotting API and standalone analysis-report entry point."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seismic.analysis import event_frequency
from seismic.cli import main


def plot_earthquake_frequency(data: pd.DataFrame, show: bool = True):
    """Plot daily earthquake frequency without changing the caller's data.

    Retains the existing date_time column and English-label correction. Return
    the Figure so notebooks and callers can save or close it explicitly.
    """
    import matplotlib.pyplot as plt

    events = data.copy()
    events["date_time"] = pd.to_datetime(events.date_time)
    daily = event_frequency(events)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily.date_time, daily.events, marker="o")
    ax.set(title="Daily Earthquake Frequency", xlabel="Date", ylabel="Number of Earthquakes")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    main(["analyze", *sys.argv[1:]])
