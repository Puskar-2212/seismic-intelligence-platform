"""Real bundled catalogue fixtures; altered copies only exercise edge cases."""

import pandas as pd
import pytest

from seismic.data import load_catalogue
from seismic.paths import DEFAULT_DATA


@pytest.fixture
def raw():
    return pd.read_csv(DEFAULT_DATA)


@pytest.fixture
def events():
    return load_catalogue().events
