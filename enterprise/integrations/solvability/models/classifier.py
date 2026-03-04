from __future__ import annotations

import base64
import pickle
from typing import Any

import numpy as np
import pandas as pd
import shap
from integrations.solvability.models.featurizer import Feature, Featurizer
from integrations.solvability.models.importance_strategy import ImportanceStrategy
from integrations.solvability.models.report import SolvabilityReport
from pydantic import (
    BaseModel,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from openhands.core.config import LLMConfig


class SolvabilityClassifier(BaseModel):
    """Machine learning pipeline for predicting the solvability of GitHub issues and similar problems.

    This classifier combines LLM-based feature extraction with traditional ML classification:
    1. Uses a Featurizer to extract semantic boolean features from issue descriptions via LLM calls
    2. Trains a RandomForestClassifier on these features to predict solvability scores
    3. Provides feature importance analysis using configurable strategies (SHAP, permutation, impurity)
    4. Generates comprehensive reports with predictions, feature analysis, and cost metrics

    The classifier supports both training on labeled data and inference on new issues, with built-in
    support for batch processing and concurrent feature extraction.
    """

    identifier: str
    """
    The identifier for the classifier.
    """

    featurizer: Featurizer
    """
    The featurizer to use for transforming the input data.
    """

    classifier: RandomForestClassifier
    """
    The RandomForestClassifier used for predicting solvability from extracted features.

    This ensemble model provides robust predictions and built-in feature importance metrics.
    """

    importance_strategy: ImportanceStrategy = ImportanceStrategy.IMPURITY
    """
    Strategy to use for calculating feature importance.
    """

    samples: int = 10
    """
    Number of samples to use for calculating feature embedding coefficients.
    """

    random_state: int | None = None
    """
    Random state for reproducibility.
    """

    _classifier_attrs: dict[str, Any] = PrivateAttr(default_factory=dict)
    """
    Private dictionary storing cached results from feature extraction and importance calculations.

    Contains keys like 'features_', 'cost_', 'feature_importances_', and 'labels_' that are populated
    during transform(), fit(), and predict() operations. Access these via the corresponding properties.

    This field is never serialized, so cached values will not persist across model save/load cycles.
    """

    model_config = {
        'arbitrary_types_allowed': True,
    }

    @model_validator(mode='after')
    def validate_random_state(self) -> SolvabilityClassifier:
        """Validate the random state configuration between this object and the classifier.
        """
        # If both random states are set, they definitely need to agree.
        if self.random_state is not None and self.classifier.random_state is not None:
            if self.random_state != self.classifier.random_state:
                raise ValueError(
                    'The random state of the classifier and the top-level classifier must agree.'
                )

        # Otherwise, we'll always set the classifier's random state to the top-level one.
        self.classifier.random_state = self.random_state

        return self

    @property
    def features_(self) -> pd.DataFrame:
        """Get the features used by the classifier for the most recent inputs.
        """
        if 'features_' not in self._classifier_attrs:
            raise ValueError(
                'SolvabilityClassifier.transform() has not yet been called.'
            )
        return self._classifier_attrs['features_']

    @property
    def cost_(self) -> pd.DataFrame:
        """Get the cost of the classifier for the most recent inputs.
        """
        if 'cost_' not in self._classifier_attrs:
            raise ValueError(
                'SolvabilityClassifier.transform() has not yet been called.'
            )
        return self._classifier_attrs['cost_']

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get the feature importances for the most recent inputs.
        """
        if 'feature_importances_' not in self._classifier_attrs:
            raise ValueError(
                'No SolvabilityClassifier methods that produce feature importances (.fit(), .predict_proba(), and '
                '.predict()) have been called.'
            )
        return self._classifier_attrs['feature_importances_']  # type: ignore[no-any-return]

    @property
    def is_fitted(self) -> bool:
        """Check if the classifier is fitted.
        """
        try:
            check_is_fitted(self.classifier)
            return True
        except NotFittedError:
            return False

    def transform(self, issues: pd.Series, llm_config: LLMConfig) -> pd.DataFrame:
        """Transform the input issues using the featurizer to extract features.

        This method orchestrates the feature extraction pipeline:
        1. Uses the featurizer to generate embeddings for all issues
        2. Converts embeddings to a structured DataFrame
        3. Separates feature columns from metadata columns
        4. Stores results for later access via properties

        Args:
            issues: A pandas Series containing the issue descriptions.
            llm_config: LLM configuration to use for feature extraction.

        Returns:
            pd.DataFrame: A DataFrame containing only the feature columns (no metadata).
        """
        # Generate feature embeddings for all issues using batch processing
        feature_embeddings = self.featurizer.embed_batch(
            issues, samples=self.samples, llm_config=llm_config
        )
        df = pd.DataFrame(embedding.to_row() for embedding in feature_embeddings)

        # Split into feature columns (used by classifier) and cost columns (metadata)
        feature_columns = [feature.identifier for feature in self.featurizer.features]
        cost_columns = [col for col in df.columns if col not in feature_columns]

        # Store both sets for access via properties
        self._classifier_attrs['features_'] = df[feature_columns]
        self._classifier_attrs['cost_'] = df[cost_columns]

        return self.features_

    def fit(
        self, issues: pd.Series, labels: pd.Series, llm_config: LLMConfig
    ) -> SolvabilityClassifier:
        """Fit the classifier to the input issues and labels.

        Args:
            issues: A pandas Series containing the issue descriptions.

            labels: A pandas Series containing the labels (0 or 1) for each issue.

            llm_config: LLM configuration to use for feature extraction.

        Returns:
            SolvabilityClassifier: The fitted classifier.
        """
        features = self.transform(issues, llm_config=llm_config)
        self.classifier.fit(features, labels)

        # Store labels for permutation importance calculation
        self._classifier_attrs['labels_'] = labels
        self._classifier_attrs['feature_importances_'] = self._importance(
            features, self.classifier.predict_proba(features), labels
        )

        return self

    def predict_proba(self, issues: pd.Series, llm_config: LLMConfig) -> np.ndarray:
        """Predict the solvability probabilities for the input issues.

        Returns class probabilities where the second column represents the probability
        of the issue being solvable (positive class).

        Args:
            issues: A pandas Series containing the issue descriptions.
            llm_config: LLM configuration to use for feature extraction.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with probabilities for each class.
                       Column 0: probability of not solvable, Column 1: probability of solvable.
        """
        features = self.transform(issues, llm_config=llm_config)
        scores = self.classifier.predict_proba(features)

        # Calculate feature importances based on the configured strategy
        # For permutation importance, we need ground truth labels if available
        labels = self._classifier_attrs.get('labels_')
        if (
            self.importance_strategy == ImportanceStrategy.PERMUTATION
            and labels is not None
        ):
            self._classifier_attrs['feature_importances_'] = self._importance(
                features, scores, labels
            )
        else:
            self._classifier_attrs['feature_importances_'] = self._importance(
                features, scores
            )

        return scores  # type: ignore[no-any-return]

    def predict(self, issues: pd.Series, llm_config: LLMConfig) -> np.ndarray:
        """Predict the solvability of the input issues by returning binary labels.

        Uses a 0.5 probability threshold to convert probabilities to binary predictions.

        Args:
            issues: A pandas Series containing the issue descriptions.
            llm_config: LLM configuration to use for feature extraction.

        Returns:
            np.ndarray: Boolean array where True indicates the issue is predicted as solvable.
        """
        probabilities = self.predict_proba(issues, llm_config=llm_config)
        # Apply 0.5 threshold to convert probabilities to binary predictions
        labels = probabilities[:, 1] >= 0.5
        return labels

    def _importance(
        self,
        features: pd.DataFrame,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate feature importance scores using the configured strategy.

        Different strategies provide different interpretations:
        - SHAP: Shapley values indicating contribution to individual predictions
        - PERMUTATION: Decrease in model performance when feature is shuffled
        - IMPURITY: Gini impurity decrease from splits on each feature

        Args:
            features: Feature matrix used for predictions.
            scores: Model prediction scores (unused for some strategies).
            labels: Ground truth labels (required for permutation importance).

        Returns:
            np.ndarray: Feature importance scores, one per feature.
        """
        match self.importance_strategy:
            case ImportanceStrategy.SHAP:
                # Use SHAP TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(self.classifier)
                shap_values = explainer.shap_values(features)
                # Return mean SHAP values for the positive class (solvable)
                return shap_values.mean(axis=0)[:, 1]  # type: ignore[no-any-return]

            case ImportanceStrategy.PERMUTATION:
                # Permutation importance requires ground truth labels
                if labels is None:
                    raise ValueError('Labels are required for permutation importance')
                result = permutation_importance(
                    self.classifier,
                    features,
                    labels,
                    n_repeats=10,  # Number of permutation rounds for stability
                    random_state=self.random_state,
                )
                return result.importances_mean  # type: ignore[no-any-return]

            case ImportanceStrategy.IMPURITY:
                # Use built-in feature importances from RandomForest
                return self.classifier.feature_importances_  # type: ignore[no-any-return]

            case _:
                raise ValueError(
                    f'Unknown importance strategy: {self.importance_strategy}'
                )

    def add_features(self, features: list[Feature]) -> SolvabilityClassifier:
        """Add new features to the classifier's featurizer.

        Note: Adding features after training requires retraining the classifier
        since the feature space will have changed.

        Args:
            features: List of Feature objects to add.

        Returns:
            SolvabilityClassifier: Self for method chaining.
        """
        for feature in features:
            if feature not in self.featurizer.features:
                self.featurizer.features.append(feature)
        return self

    def forget_features(self, features: list[Feature]) -> SolvabilityClassifier:
        """Remove features from the classifier's featurizer.

        Note: Removing features after training requires retraining the classifier
        since the feature space will have changed.

        Args:
            features: List of Feature objects to remove.

        Returns:
            SolvabilityClassifier: Self for method chaining.
        """
        for feature in features:
            try:
                self.featurizer.features.remove(feature)
            except ValueError:
                # Feature not in list, continue with others
                continue
        return self

    @field_serializer('classifier')
    @staticmethod
    def _rfc_to_json(rfc: RandomForestClassifier) -> str:
        """Convert a RandomForestClassifier to a JSON-compatible value (a string).
        """
        return base64.b64encode(pickle.dumps(rfc)).decode('utf-8')

    @field_validator('classifier', mode='before')
    @staticmethod
    def _json_to_rfc(value: str | RandomForestClassifier) -> RandomForestClassifier:
        """Convert a JSON-compatible value (a string) back to a RandomForestClassifier.
        """
        if isinstance(value, RandomForestClassifier):
            return value

        if isinstance(value, str):
            try:
                model = pickle.loads(base64.b64decode(value))
                if isinstance(model, RandomForestClassifier):
                    return model
            except Exception as e:
                raise ValueError(f'Failed to decode the classifier: {e}')

        raise ValueError(
            'The classifier must be a RandomForestClassifier or a JSON-compatible dictionary.'
        )

    def solvability_report(
        self, issue: str, llm_config: LLMConfig, **kwargs: Any
    ) -> SolvabilityReport:
        """Generate a solvability report for the given issue.

        Args:
            issue: The issue description for which to generate the report.
            llm_config: Optional LLM configuration to use for feature extraction.
            kwargs: Additional metadata to include in the report.

        Returns:
            SolvabilityReport: The generated solvability report.
        """
        if not self.is_fitted:
            raise ValueError(
                'The classifier must be fitted before generating a report.'
            )

        scores = self.predict_proba(pd.Series([issue]), llm_config=llm_config)

        return SolvabilityReport(
            identifier=self.identifier,
            issue=issue,
            score=scores[0, 1],
            features=self.features_.iloc[0].to_dict(),
            samples=self.samples,
            importance_strategy=self.importance_strategy,
            # Unlike the features, the importances are just a series with no link
            # to the actual feature names. For that we have to recombine with the
            # feature identifiers.
            feature_importances=dict(
                zip(
                    self.featurizer.feature_identifiers(),
                    self.feature_importances_.tolist(),
                )
            ),
            random_state=self.random_state,
            metadata=dict(kwargs) if kwargs else None,
            # Both cost and response_latency are columns in the cost_ DataFrame,
            # so we can get both by just unpacking the first row.
            **self.cost_.iloc[0].to_dict(),
        )

    def __call__(
        self, issue: str, llm_config: LLMConfig, **kwargs: Any
    ) -> SolvabilityReport:
        """Generate a solvability report for the given issue.
        """
        return self.solvability_report(issue, llm_config=llm_config, **kwargs)
