"""Store for managing verified LLM models in the database."""

from dataclasses import dataclass

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Identity,
    Integer,
    String,
    UniqueConstraint,
    and_,
    func,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession
from storage.base import Base

from enterprise.server.verified_models.verified_model_models import (
    VerifiedModel,
    VerifiedModelPage,
)
from openhands.app_server.config import depends_db_session
from openhands.core.logger import openhands_logger as logger


class StoredVerifiedModel(Base):  # type: ignore
    """A verified LLM model available in the model selector.

    The composite unique constraint on (model_name, provider) allows the same
    model name to exist under different providers (e.g. 'claude-sonnet' under
    both 'openhands' and 'anthropic').
    """

    __tablename__ = 'verified_models'
    __table_args__ = (
        UniqueConstraint('model_name', 'provider', name='uq_verified_model_provider'),
    )

    id = Column(Integer, Identity(), primary_key=True)
    model_name = Column(String(255), nullable=False)
    provider = Column(String(100), nullable=False, index=True)
    is_enabled = Column(
        Boolean, nullable=False, default=True, server_default=text('true')
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


def verified_model(result: StoredVerifiedModel) -> VerifiedModel:
    return VerifiedModel(
        id=result.id,
        model_name=result.model_name,
        provider=result.provider,
        is_enabled=result.is_enabled,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


@dataclass
class VerifiedModelService:
    """Store for CRUD operations on verified models.

    Follows the async pattern with db_session as an attribute.
    """

    db_session: AsyncSession

    async def search_verified_models(
        self,
        provider: str | None = None,
        enabled_only: bool = True,
        page_id: str | None = None,
        limit: int = 100,
    ) -> VerifiedModelPage:
        """Search for verified models with optional filtering and pagination.

        Args:
            provider: Optional provider name to filter by (e.g., 'openhands', 'anthropic')
            enabled_only: If True, only return enabled models (default: True)
            page_id: Page id for pagination
            limit: Maximum number of records to return

        Returns:
            SearchModelsResult containing items list and has_more flag
        """
        query = select(StoredVerifiedModel)

        # Build filters
        filters = []
        if provider:
            filters.append(StoredVerifiedModel.provider == provider)
        if enabled_only:
            filters.append(StoredVerifiedModel.is_enabled.is_(True))

        if filters:
            query = query.where(and_(*filters))

        # Order by provider, then model_name
        query = query.order_by(
            StoredVerifiedModel.provider, StoredVerifiedModel.model_name
        )

        # Fetch limit + 1 to check if there are more results
        offset = int(page_id or '0')
        query = query.offset(offset).limit(limit + 1)

        result = await self.db_session.execute(query)
        results = list(result.scalars().all())
        has_more = len(results) > limit
        next_page_id = None

        # Return only the requested number of results
        if has_more:
            next_page_id = str(offset + limit)
            results.pop()

        items = [verified_model(result) for result in results]
        return VerifiedModelPage(items=items, next_page_id=next_page_id)

    async def get_model(self, model_name: str, provider: str) -> VerifiedModel | None:
        """Get a model by its composite key (model_name, provider).

        Args:
            model_name: The model identifier
            provider: The provider name
        """
        query = select(StoredVerifiedModel).where(
            and_(
                StoredVerifiedModel.model_name == model_name,
                StoredVerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(query)
        return result.scalars().first()

    async def create_verified_model(
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
        existing_query = select(StoredVerifiedModel).where(
            and_(
                StoredVerifiedModel.model_name == model_name,
                StoredVerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(existing_query)
        existing = result.scalars().first()
        if existing:
            raise ValueError(f'Model {provider}/{model_name} already exists')

        model = StoredVerifiedModel(
            model_name=model_name,
            provider=provider,
            is_enabled=is_enabled,
        )
        self.db_session.add(model)
        await self.db_session.commit()
        await self.db_session.refresh(model)
        logger.info(f'Created verified model: {provider}/{model_name}')
        return verified_model(model)

    async def update_verified_model(
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
        query = select(StoredVerifiedModel).where(
            and_(
                StoredVerifiedModel.model_name == model_name,
                StoredVerifiedModel.provider == provider,
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
        return verified_model(model)

    async def delete_verified_model(self, model_name: str, provider: str):
        """Delete a verified model.

        Args:
            model_name: The model name to delete
            provider: The provider name

        Returns:
            True if deleted, False if not found
        """
        query = select(StoredVerifiedModel).where(
            and_(
                StoredVerifiedModel.model_name == model_name,
                StoredVerifiedModel.provider == provider,
            )
        )
        result = await self.db_session.execute(query)
        model = result.scalars().first()
        if not model:
            raise ValueError('Unknown model')

        await self.db_session.delete(model)
        await self.db_session.commit()
        logger.info(f'Deleted verified model: {provider}/{model_name}')


def verified_model_store_dependency(db_session: AsyncSession = depends_db_session()):
    return VerifiedModelService(db_session)
