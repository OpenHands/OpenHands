"""Shared utilities for marketplace registration validation and conversion."""

import logging

from pydantic import ValidationError

from openhands.storage.data_models.settings import MarketplaceRegistration

logger = logging.getLogger(__name__)


def validate_and_convert_marketplaces(
    raw_marketplaces: list[dict | MarketplaceRegistration] | None,
    source_name: str = 'marketplaces',
) -> list[MarketplaceRegistration]:
    """Validate and convert raw marketplace data to MarketplaceRegistration objects.

    This function handles the common pattern of validating marketplace data
    that comes from database storage (stored as dicts) or direct model instances.
    Invalid entries are logged and skipped (graceful degradation).

    Args:
        raw_marketplaces: List of raw marketplace data (dicts or model instances)
        source_name: Descriptive name for logging (e.g., "org", "user settings")

    Returns:
        List of validated MarketplaceRegistration objects.

    Example:
        >>> data = [{'name': 'test', 'source': 'github:owner/repo'}]
        >>> registrations = validate_and_convert_marketplaces(data, "my-org")
        >>> len(registrations)
        1
    """
    if not raw_marketplaces:
        return []

    validated = []
    for i, mp in enumerate(raw_marketplaces):
        try:
            if isinstance(mp, dict):
                validated.append(MarketplaceRegistration.model_validate(mp))
            elif isinstance(mp, MarketplaceRegistration):
                validated.append(mp)
            else:
                raise ValueError(
                    f'Expected dict or MarketplaceRegistration, got {type(mp).__name__}'
                )
        except (ValidationError, ValueError) as e:
            logger.warning(
                f'Skipping invalid marketplace at index {i} in {source_name}: {e}'
            )
            continue

    return validated
