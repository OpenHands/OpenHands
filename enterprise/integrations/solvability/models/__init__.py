"""Solvability Models Package

This package contains the core machine learning models and components for predicting
the solvability of GitHub issues and similar technical problems.

The solvability prediction system works by:
1. Using a Featurizer to extract semantic features from issue descriptions via LLM calls
2. Training a RandomForestClassifier on these features to predict solvability
3. Generating detailed reports with feature importance analysis

Key Components:
- Feature: Defines individual features that can be extracted from issues
- Featurizer: Orchestrates LLM-based feature extraction with sampling and batching
- SolvabilityClassifier: Main ML pipeline combining featurization and classification
- SolvabilityReport: Comprehensive output with predictions, feature analysis, and metadata
- ImportanceStrategy: Configurable methods for calculating feature importance (SHAP, permutation, impurity)
"""

from integrations.solvability.models.classifier import SolvabilityClassifier
from integrations.solvability.models.featurizer import (
    EmbeddingDimension,
    Feature,
    FeatureEmbedding,
    Featurizer,
)
from integrations.solvability.models.importance_strategy import ImportanceStrategy
from integrations.solvability.models.report import SolvabilityReport

__all__ = [
    'Feature',
    'EmbeddingDimension',
    'FeatureEmbedding',
    'Featurizer',
    'ImportanceStrategy',
    'SolvabilityClassifier',
    'SolvabilityReport',
]
