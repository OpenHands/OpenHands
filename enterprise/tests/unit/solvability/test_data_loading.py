"""Unit tests for data loading functionality in solvability/data.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from integrations.solvability.data import available_classifiers, load_classifier
from integrations.solvability.models.classifier import SolvabilityClassifier
from pydantic import ValidationError


def test_load_classifier_default():
    """Test loading the default classifier."""
    classifier = load_classifier('default-classifier')

    assert isinstance(classifier, SolvabilityClassifier)
    assert classifier.identifier == 'default-classifier'
    assert classifier.featurizer is not None
    assert classifier.classifier is not None


def test_load_classifier_not_found():
    """Test loading a non-existent classifier raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_classifier('non-existent-classifier')

    assert "Classifier 'non-existent-classifier' not found" in str(exc_info.value)


def test_available_classifiers():
    """Test listing available classifiers."""
    classifiers = available_classifiers()

    assert isinstance(classifiers, list)
    assert 'default-classifier' in classifiers
    assert len(classifiers) >= 1


def test_load_classifier_with_mock_data(solvability_classifier):
    """Test loading a classifier with mocked data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'test-classifier.json'

        with test_file.open('w') as f:
            f.write(solvability_classifier.model_dump_json())

        with patch('integrations.solvability.data.Path') as mock_path:
            mock_path.return_value.parent = Path(tmpdir)

            classifier = load_classifier('test-classifier')

            assert isinstance(classifier, SolvabilityClassifier)
            assert classifier.identifier == 'test-classifier'


def test_available_classifiers_with_mock_directory():
    """Test listing classifiers in a mocked directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        (tmpdir_path / 'classifier1.json').touch()
        (tmpdir_path / 'classifier2.json').touch()
        (tmpdir_path / 'not-a-json.txt').touch()

        with patch('integrations.solvability.data.Path') as mock_path:
            mock_path.return_value.parent = tmpdir_path

            classifiers = available_classifiers()

            assert len(classifiers) == 2
            assert 'classifier1' in classifiers
            assert 'classifier2' in classifiers
            assert 'not-a-json' not in classifiers


def test_load_classifier_invalid_json():
    """Test loading a classifier with invalid JSON content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'invalid-classifier.json'

        with test_file.open('w') as f:
            f.write('{ invalid json content')

        with patch('integrations.solvability.data.Path') as mock_path:
            mock_path.return_value.parent = Path(tmpdir)

            with pytest.raises(ValidationError):
                load_classifier('invalid-classifier')


def test_load_classifier_valid_json_invalid_schema():
    """Test loading a classifier with valid JSON but invalid schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'invalid-schema.json'

        with test_file.open('w') as f:
            json.dump({'not': 'a valid classifier'}, f)

        with patch('integrations.solvability.data.Path') as mock_path:
            mock_path.return_value.parent = Path(tmpdir)

            with pytest.raises(ValidationError):
                load_classifier('invalid-schema')


def test_available_classifiers_empty_directory():
    """Test listing classifiers in an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('integrations.solvability.data.Path') as mock_path:
            mock_path.return_value.parent = Path(tmpdir)

            classifiers = available_classifiers()

            assert classifiers == []


def test_load_classifier_path_construction():
    """Test that the classifier path is constructed correctly."""
    with patch('integrations.solvability.data.Path') as mock_path:
        mock_parent = mock_path.return_value.parent
        mock_parent.__truediv__.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            load_classifier('test-name')

        mock_parent.__truediv__.assert_called_once_with('test-name.json')
