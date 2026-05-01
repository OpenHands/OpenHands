import os
from pathlib import Path

__package_name__ = 'openhands_ai'


def get_version():
    # Try getting the version from pyproject.toml
    try:
        root_dir = Path(os.path.abspath(__file__)).parents[2]
        candidate_paths = [
            root_dir / 'pyproject.toml',
            root_dir / 'openhands' / 'pyproject.toml',
        ]
        for file_path in candidate_paths:
            if file_path.is_file():
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('version ='):
                            return line.split('=', 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass

    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(__package_name__)
    except (ImportError, PackageNotFoundError):
        pass

    try:
        from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

        try:
            return get_distribution(__package_name__).version
        except DistributionNotFound:
            pass
    except ImportError:
        pass

    return 'unknown'


try:
    __version__ = get_version()
except Exception:
    __version__ = 'unknown'
