"""Repository paths, independent of the process working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed_data" / "earthquake_data.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"


def project_path(path: str | Path) -> Path:
    """Resolve relative CLI paths against the project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
