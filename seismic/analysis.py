"""Descriptive temporal and sequence analyses of the observed catalogue."""

import pandas as pd


def event_frequency(data: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    """Count events in calendar bins (edge bins may have partial coverage)."""
    if frequency not in {"h", "D", "W", "MS", "YS"}:
        raise ValueError("Frequency must be h, D, W, MS or YS.")
    return (data.set_index("date_time").resample(frequency).size()
            .rename("events").reset_index())


def temporal_summary(data: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Daily counts and time-based rolling activity; not a swarm classifier."""
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    ordered = data.sort_values("date_time").set_index("date_time")
    daily = ordered.resample("D").agg(events=("mag", "size"), magnitude_sum=("mag", "sum"))
    daily["rolling_events"] = daily.events.rolling(f"{window_days}D").sum()
    total_mag = daily.magnitude_sum.rolling(f"{window_days}D").sum()
    daily["rolling_mean_magnitude"] = total_mag / daily.rolling_events.replace(0, float("nan"))
    return daily.drop(columns="magnitude_sum").reset_index()


def event_sequences(data: pd.DataFrame) -> pd.DataFrame:
    """Event intervals in chronological order; ties have zero elapsed time."""
    ordered = data.sort_values("date_time", kind="stable").copy()
    ordered["hours_since_previous"] = ordered.date_time.diff().dt.total_seconds() / 3600
    return ordered


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data[["mag", "depth"]].describe()


def meaningful_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation only for measured, varying columns with enough pairs."""
    numeric = data[["mag", "depth"]]
    varying = [c for c in numeric if numeric[c].nunique() > 1]
    return numeric[varying].corr(min_periods=3)


def strongest_events(data: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    return data.nlargest(count, "mag")[["event_id", "date_time", "mag", "depth", "location"]]
