"""Store for managing verified LLM models in the database."""

from typing import TypedDict

from sqlalchemy import and_
from storage.database import session_maker
from storage.verified_model import VerifiedModel

from openhands.core.logger import openhands_logger as logger


class SearchModelsResult(TypedDict):
    """Result of search_models with pagination info."""

    items: list[VerifiedModel]
    has_more: bool


class VerifiedModelStore:
    """Store for CRUD operations on verified models.

    Follows the project convention of static methods with session_maker()
    (see UserStore, OrgMemberStore for reference).
    """

    @staticmethod
    def search_models(
        provider: str | None = None,
        enabled_only: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> SearchModelsResult:
        """Search for verified models with optional filtering and pagination.

        Args:
            provider: Optional provider name to filter by (e.g., 'openhands', 'anthropic')
            enabled_only: If True, only return enabled models (default: True)
            offset: Number of records to skip (for pagination)
            limit: Maximum number of records to return

        Returns:
            SearchModelsResult containing items list and has_more flag
        """
        with session_maker() as session:
            query = session.query(VerifiedModel)

            # Build filters
            filters = []
            if provider:
                filters.append(VerifiedModel.provider == provider)
            if enabled_only:
                filters.append(VerifiedModel.is_enabled.is_(True))

            if filters:
                query = query.filter(and_(*filters))

            # Order by provider, then model_name
            query = query.order_by(VerifiedModel.provider, VerifiedModel.model_name)

            # Fetch limit + 1 to check if there are more results
            results = query.offset(offset).limit(limit + 1).all()
            has_more = len(results) > limit

            # Return only the requested number of results
            if has_more:
                results = results[:limit]

            return SearchModelsResult(items=results, has_more=has_more)

    @staticmethod
    def get_enabled_models() -> list[VerifiedModel]:
        """Get all enabled models.

        Returns:
            list[VerifiedModel]: All models where is_enabled is True
        """
        result = VerifiedModelStore.search_models(enabled_only=True, limit=1000)
        return result['items']

    @staticmethod
    def get_models_by_provider(provider: str) -> list[VerifiedModel]:
        """Get all enabled models for a specific provider.

        Args:
            provider: The provider name (e.g., 'openhands', 'anthropic')

        Note:
            This method is deprecated. Use search_models() instead.
        """
        result = VerifiedModelStore.search_models(provider=provider, enabled_only=True, limit=1000)
        return result['items']

    @staticmethod
    def get_all_models() -> list[VerifiedModel]:
        """Get all models (including disabled).

        Note:
            This method is deprecated. Use search_models() instead.
        """
        result = VerifiedModelStore.search_models(enabled_only=False, limit=1000)
        return result['items']

    @staticmethod
    def get_model(model_name: str, provider: str) -> VerifiedModel | None:
        """Get a model by its composite key (model_name, provider).

        Args:
            model_name: The model identifier
            provider: The provider name
        """
        with session_maker() as session:
            return (
                session.query(VerifiedModel)
                .filter(
                    and_(
                        VerifiedModel.model_name == model_name,
                        VerifiedModel.provider == provider,
                    )
                )
                .first()
            )

    @staticmethod
    def create_model(
        model_name: str, provider: str, is_enabled: bool = True
    ) -> VerifiedModel:
        """Create a new verified model.

        Args:
            model_name: The model identifier
            provider: The provider name
            is_enabled: Whether the model is enabled (default True)

        Raises:
            ValueError: If a model with the same (model_name, provider) already exists
        """
        with session_maker() as session:
            existing = (
                session.query(VerifiedModel)
                .filter(
                    and_(
                        VerifiedModel.model_name == model_name,
                        VerifiedModel.provider == provider,
                    )
                )
                .first()
            )
            if existing:
                raise ValueError(f'Model {provider}/{model_name} already exists')

            model = VerifiedModel(
                model_name=model_name,
                provider=provider,
                is_enabled=is_enabled,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            logger.info(f'Created verified model: {provider}/{model_name}')
            return model

    @staticmethod
    def update_model(
        model_name: str,
        provider: str,
        is_enabled: bool | None = None,
    ) -> VerifiedModel | None:
        """Update an existing verified model.

        Args:
            model_name: The model name to update
            provider: The provider name
            is_enabled: New enabled state (optional)

        Returns:
            The updated model if found, None otherwise
        """
        with session_maker() as session:
            model = (
                session.query(VerifiedModel)
                .filter(
                    and_(
                        VerifiedModel.model_name == model_name,
                        VerifiedModel.provider == provider,
                    )
                )
                .first()
            )
            if not model:
                return None

            if is_enabled is not None:
                model.is_enabled = is_enabled

            session.commit()
            session.refresh(model)
            logger.info(f'Updated verified model: {provider}/{model_name}')
            return model

    @staticmethod
    def delete_model(model_name: str, provider: str) -> bool:
        """Delete a verified model.

        Args:
            model_name: The model name to delete
            provider: The provider name

        Returns:
            True if deleted, False if not found
        """
        with session_maker() as session:
            model = (
                session.query(VerifiedModel)
                .filter(
                    and_(
                        VerifiedModel.model_name == model_name,
                        VerifiedModel.provider == provider,
                    )
                )
                .first()
            )
            if not model:
                return False

            session.delete(model)
            session.commit()
            logger.info(f'Deleted verified model: {provider}/{model_name}')
            return True
