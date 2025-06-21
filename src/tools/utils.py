from pathlib import Path


def create_directory(path: str):
    """Create a directory if it does not exist."""

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
