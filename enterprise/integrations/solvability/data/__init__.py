"""Utilities for loading and managing pre-trained classifiers.

Assumes that classifiers are stored adjacent to this file in the `solvability/data` directory, using a simple
`name + .json` pattern.
"""

from pathlib import Path

from integrations.solvability.models.classifier import SolvabilityClassifier


def load_classifier(name: str) -> SolvabilityClassifier:
    """Load a classifier by name.

    Args:
        name (str): The name of the classifier to load.

    Returns:
        SolvabilityClassifier: The loaded classifier instance.
    """
    data_dir = Path(__file__).parent
    classifier_path = data_dir / f'{name}.json'

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier '{name}' not found at {classifier_path}")

    with classifier_path.open('r') as f:
        return SolvabilityClassifier.model_validate_json(f.read())


def available_classifiers() -> list[str]:
    """List all available classifiers in the data directory.

    Returns:
        list[str]: A list of classifier names (without the .json extension).
    """
    data_dir = Path(__file__).parent
    return [f.stem for f in data_dir.glob('*.json') if f.is_file()]
