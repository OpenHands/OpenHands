"""Store for managing verified LLM models in the database."""

from dataclasses import dataclass

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.database import a_session_maker
from storage.verified_model import VerifiedModel

from openhands.core.logger import openhands_logger as logger


@dataclass
class SearchModelsResult:
    """Result of search_models with pagination info."""

    items: list[VerifiedModel]
    has_more: bool


class VerifiedModelStore:
    """Store for CRUD operations on verified models.

    Follows the async pattern with db_session as an attribute.
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize the store with a database session.

        Args:
            db_session: The async database session to use for queries.
        """
        self.db_session = db_session

    async def search_models(
        self,
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
        query = select(VerifiedModel)

        # Build filters
        filters = []
        if provider:
            filters.append(VerifiedModel.provider == provider)
        if enabled_only:
            filters.append(VerifiedModel.is_enabled.is_(True))

        if filters:
            query = query.where(and_(*filters))

        # Order by provider, then model_name
        query = query.order_by(VerifiedModel.provider, VerifiedModel.model_name)

        # Fetch limit + 1 to check if there are more results
        query = query.offset(offset).limit(limit + 1)

        result = await self.db_session.execute(query)
        results = list(result.scalars().all())
        has_more = len(results) > limit

        # Return only the requested number of results
        if has_more:
            results = results[:limit]

        return SearchModelsResult(items=results, has_more=has_more)

    async def get_enabled_models(self) -> list[VerifiedModel]:
        """Get all enabled models.

        Returns:
            list[VerifiedModel]: All models where is_enabled is True
        """
        # Fetch all enabled models without limit to avoid silent data loss
        result = await self.search_models(enabled_only=True, limit=10**9)
        return result.items

    async def get_models_by_provider(self, provider: str) -> list[VerifiedModel]:
        """Get all enabled models for a specific provider.

        Args:
            provider: The provider name (e.g., 'openhands', 'anthropic')

        .. deprecated::
            Use :meth:`search_models` instead for SQL-level filtering.
        """
        result = await self.search_models(provider=provider, enabled_only=True, limit=10**9)
        return result.items

    async def get_all_models(self) -> list[VerifiedModel]:
        """Get all models (including disabled).

        .. deprecated::
            Use :meth:`search_models` instead for SQL-level filtering.
        """
        result = await self.search_models(enabled_only=False, limit=10**9)
        return result.items

    async def get_model(self, model_name: str, provider: str) -> VerifiedModel | None:
        """Get a model by its composite key (model_name, provider).

        Args:
            model_name: The model identifier
            provider: The provider name
        """
        query = select(VerifiedModel).where(
            and_(
                VerifiedModel.model_name == model_name,
                VerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(query)
        return result.scalars().first()

    async def create_model(
        self,
        model_name: str,
        provider: str,
        is_enabled: bool = True,
    ) -> VerifiedModel:
        """Create a new verified model.

        Args:
            model_name: The model identifier
            provider: The provider name
            is_enabled: Whether the model is enabled (default True)

        Raises:
            ValueError: If a model with the same (model_name, provider) already exists
        """
        existing_query = select(VerifiedModel).where(
            and_(
                VerifiedModel.model_name == model_name,
                VerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(existing_query)
        existing = result.scalars().first()
        if existing:
            raise ValueError(f'Model {provider}/{model_name} already exists')

        model = VerifiedModel(
            model_name=model_name,
            provider=provider,
            is_enabled=is_enabled,
        )
        self.db_session.add(model)
        await self.db_session.commit()
        await self.db_session.refresh(model)
        logger.info(f'Created verified model: {provider}/{model_name}')
        return model

    async def update_model(
        self,
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
        query = select(VerifiedModel).where(
            and_(
                VerifiedModel.model_name == model_name,
                VerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(query)
        model = result.scalars().first()
        if not model:
            return None

        if is_enabled is not None:
            model.is_enabled = is_enabled

        await self.db_session.commit()
        await self.db_session.refresh(model)
        logger.info(f'Updated verified model: {provider}/{model_name}')
        return model

    async def delete_model(self, model_name: str, provider: str) -> bool:
        """Delete a verified model.

        Args:
            model_name: The model name to delete
            provider: The provider name

        Returns:
            True if deleted, False if not found
        """
        query = select(VerifiedModel).where(
            and_(
                VerifiedModel.model_name == model_name,
                VerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(query)
        model = result.scalars().first()
        if not model:
            return False

        await self.db_session.delete(model)
        await self.db_session.commit()
        logger.info(f'Deleted verified model: {provider}/{model_name}')
        return True


# Module-level async convenience functions for backward compatibility
async def search_models(
    provider: str | None = None,
    enabled_only: bool = True,
    offset: int = 0,
    limit: int = 100,
) -> SearchModelsResult:
    """Search for verified models (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.search_models(provider, enabled_only, offset, limit)


async def get_enabled_models() -> list[VerifiedModel]:
    """Get all enabled models (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.get_enabled_models()


async def get_models_by_provider(provider: str) -> list[VerifiedModel]:
    """Get all enabled models for a specific provider (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.get_models_by_provider(provider)


async def get_all_models() -> list[VerifiedModel]:
    """Get all models (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.get_all_models()


async def get_model(model_name: str, provider: str) -> VerifiedModel | None:
    """Get a model by its composite key (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.get_model(model_name, provider)


async def create_model(
    model_name: str,
    provider: str,
    is_enabled: bool = True,
) -> VerifiedModel:
    """Create a new verified model (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.create_model(model_name, provider, is_enabled)


async def update_model(
    model_name: str,
    provider: str,
    is_enabled: bool | None = None,
) -> VerifiedModel | None:
    """Update an existing verified model (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.update_model(model_name, provider, is_enabled)


async def delete_model(model_name: str, provider: str) -> bool:
    """Delete a verified model (module-level async convenience function)."""
    async with a_session_maker() as session:
        store = VerifiedModelStore(session)
        return await store.delete_model(model_name, provider)
