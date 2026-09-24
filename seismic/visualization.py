"""Shared Plotly figures for the dashboard and standalone HTML reports."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from seismic.analysis import event_frequency, meaningful_correlation

LABELS = {"mag": "Magnitude", "depth": "Depth (km)", "date_time": "Recorded time",
          "events": "Recorded events", "location": "Location"}


def style(fig: go.Figure) -> go.Figure:
    return fig.update_layout(template="plotly_white", margin=dict(l=30, r=25, t=65, b=40))


def frequency_chart(data: pd.DataFrame, frequency: str = "D") -> go.Figure:
    return style(px.bar(event_frequency(data, frequency), x="date_time", y="events",
                        title="Recorded earthquake frequency", labels=LABELS))


def distribution_chart(data: pd.DataFrame, field: str) -> go.Figure:
    return style(px.histogram(data, x=field, nbins=20, labels=LABELS,
                              title=f"{LABELS[field]} distribution"))


def magnitude_depth_chart(data: pd.DataFrame) -> go.Figure:
    return style(px.scatter(data, x="mag", y="depth", color="mag",
                            hover_data=["date_time", "location"], labels=LABELS,
                            title="Magnitude and measured depth", color_continuous_scale="Viridis"))


def category_chart(data: pd.DataFrame, field: str, title: str) -> go.Figure:
    values = data[field]
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype("string")
    counts = values.fillna("Unknown")
    counts = counts.value_counts(sort=False).rename_axis(field).reset_index(name="events")
    return style(px.bar(counts, x=field, y="events", title=title, labels=LABELS))


def correlation_chart(data: pd.DataFrame) -> go.Figure:
    corr = meaningful_correlation(data)
    if len(corr) < 2:
        return style(go.Figure().add_annotation(text="Insufficient varying paired measurements", showarrow=False))
    return style(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                           text_auto=".2f", title="Magnitude–depth Pearson correlation (descriptive)"))


def earthquake_map(data: pd.DataFrame) -> go.Figure | None:
    """Map only genuine coordinate pairs; return None when none are available."""
    located = data.dropna(subset=["latitude", "longitude"]).copy()
    if located.empty:
        return None
    located["marker_size"] = np.maximum(located.mag + 2, 0.5)
    return style(px.scatter_mapbox(
        located, lat="latitude", lon="longitude", size="marker_size", color="mag",
        hover_name="location", hover_data={"mag": True, "depth": True, "date_time": True,
                                           "marker_size": False},
        mapbox_style="open-street-map", zoom=3, size_max=22, labels=LABELS,
        title="Recorded earthquake locations", color_continuous_scale="Viridis",
    ))


def eda_figures(data: pd.DataFrame) -> dict[str, go.Figure]:
    figures = {
        "frequency": frequency_chart(data),
        "magnitude_distribution": distribution_chart(data, "mag"),
        "depth_distribution": distribution_chart(data, "depth"),
        "magnitude_depth": magnitude_depth_chart(data),
        "monthly_counts": category_chart(data, "month", "Recorded events by month (not seasonality)"),
        "magnitude_categories": category_chart(data, "magnitude_category", "Magnitude categories"),
        "depth_categories": category_chart(data, "depth_category", "Depth categories"),
        "correlation": correlation_chart(data),
    }
    if data.year.nunique() > 1:
        figures["yearly_counts"] = frequency_chart(data, "YS").update_layout(title="Recorded events by year")
    spatial = earthquake_map(data)
    if spatial is not None:
        figures["earthquake_map"] = spatial
    return figures
