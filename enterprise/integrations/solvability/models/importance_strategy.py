from enum import Enum


class ImportanceStrategy(str, Enum):
    """Strategy to use for calculating feature importances, which are used to estimate the predictive power of each feature
    in training loops and explanations.
    """

    SHAP = 'shap'
    """
    Use SHAP (SHapley Additive exPlanations) to calculate feature importances.
    """

    PERMUTATION = 'permutation'
    """
    Use the permutation-based feature importances.
    """

    IMPURITY = 'impurity'
    """
    Use the impurity-based feature importances from the RandomForestClassifier.
    """
