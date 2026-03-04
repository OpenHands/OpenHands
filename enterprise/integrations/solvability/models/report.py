from datetime import datetime
from typing import Any

from integrations.solvability.models.importance_strategy import ImportanceStrategy
from pydantic import BaseModel, Field


class SolvabilityReport(BaseModel):
    """Comprehensive report containing solvability predictions and analysis for a single issue.

    This report includes the solvability score, extracted feature values, feature importance analysis,
    cost metrics (tokens and latency), and metadata about the prediction process. It serves as the
    primary output format for solvability analysis and can be used for logging, debugging, and
    generating human-readable summaries.
    """

    identifier: str
    """
    The identifier of the solvability model used to generate the report.
    """

    issue: str
    """
    The issue description for which the solvability is predicted.

    This field is exactly the input to the solvability model.
    """

    score: float
    """
    [0, 1]-valued score indicating the likelihood of the issue being solvable.
    """

    prompt_tokens: int
    """
    Total number of prompt tokens used in API calls made to generate the features.
    """

    completion_tokens: int
    """
    Total number of completion tokens used in API calls made to generate the features.
    """

    response_latency: float
    """
    Total response latency of API calls made to generate the features.
    """

    features: dict[str, float]
    """
    [0, 1]-valued scores for each feature in the model.

    These are the values fed to the random forest classifier to generate the solvability score.
    """

    samples: int
    """
    Number of samples used to compute the feature embedding coefficients.
    """

    importance_strategy: ImportanceStrategy
    """
    Strategy used to calculate feature importances.
    """

    feature_importances: dict[str, float]
    """
    Importance scores for each feature in the model.

    Interpretation of these scores depends on the importance strategy used.
    """

    created_at: datetime = Field(default_factory=datetime.now)
    """
    Datetime when the report was created.
    """

    random_state: int | None = None
    """
    Classifier random state used when generating this report.
    """

    metadata: dict[str, Any] | None = None
    """
    Metadata for logging and debugging purposes.
    """
