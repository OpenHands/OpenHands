"""Store for managing verified LLM models in the database."""

from sqlalchemy import and_
from storage.database import session_maker
from storage.verified_model import VerifiedModel

from openhands.core.logger import openhands_logger as logger


class VerifiedModelStore:
    """Store for CRUD operations on verified models.

    Follows the project convention of static methods with session_maker()
    (see UserStore, OrgMemberStore for reference).
    """

    @staticmethod
    def get_enabled_models() -> list[VerifiedModel]:
        """Get all enabled models.

        Returns:
            list[VerifiedModel]: All models where is_enabled is True
        """
        with session_maker() as session:
            return (
                session.query(VerifiedModel)
                .filter(VerifiedModel.is_enabled.is_(True))
                .order_by(VerifiedModel.provider, VerifiedModel.model_name)
                .all()
            )

    @staticmethod
    def get_models_by_provider(provider: str) -> list[VerifiedModel]:
        """Get all enabled models for a specific provider.

        Args:
            provider: The provider name (e.g., 'openhands', 'anthropic')
        """
        with session_maker() as session:
            return (
                session.query(VerifiedModel)
                .filter(
                    and_(
                        VerifiedModel.provider == provider,
                        VerifiedModel.is_enabled.is_(True),
                    )
                )
                .order_by(VerifiedModel.model_name)
                .all()
            )

    @staticmethod
    def get_all_models() -> list[VerifiedModel]:
        """Get all models (including disabled)."""
        with session_maker() as session:
            return (
                session.query(VerifiedModel)
                .order_by(VerifiedModel.provider, VerifiedModel.model_name)
                .all()
            )

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
