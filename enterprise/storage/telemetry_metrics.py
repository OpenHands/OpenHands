"""SQLAlchemy model for telemetry metrics data.

This model stores individual metric collection records with upload tracking
and retry logic for the OpenHands Enterprise Telemetry Service.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from storage.base import Base


class TelemetryMetrics(Base):  # type: ignore
    """Stores collected telemetry metrics with upload tracking.

    Each record represents a single metrics collection event with associated
    metadata for upload status and retry logic.
    """

    __tablename__ = 'telemetry_metrics'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    collected_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
    metrics_data = Column(JSON, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    upload_attempts = Column(Integer, nullable=False, default=0)
    last_upload_error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    def __init__(
        self,
        metrics_data: Dict[str, Any],
        collected_at: Optional[datetime] = None,
        **kwargs,
    ):
        """Initialize a new telemetry metrics record.

        Args:
            metrics_data: Dictionary containing the collected metrics
            collected_at: Timestamp when metrics were collected (defaults to now)
            **kwargs: Additional keyword arguments for SQLAlchemy
        """
        super().__init__(**kwargs)

        # Set defaults for fields that would normally be set by SQLAlchemy
        now = datetime.now(UTC)
        if not hasattr(self, 'id') or self.id is None:
            self.id = str(uuid.uuid4())
        if not hasattr(self, 'upload_attempts') or self.upload_attempts is None:
            self.upload_attempts = 0
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = now
        if not hasattr(self, 'updated_at') or self.updated_at is None:
            self.updated_at = now

        self.metrics_data = metrics_data
        if collected_at:
            self.collected_at = collected_at
        elif not hasattr(self, 'collected_at') or self.collected_at is None:
            self.collected_at = now

    def mark_uploaded(self) -> None:
        """Mark this metrics record as successfully uploaded."""
        self.uploaded_at = datetime.now(UTC)
        self.last_upload_error = None

    def mark_upload_failed(self, error_message: str) -> None:
        """Mark this metrics record as having failed upload.

        Args:
            error_message: Description of the upload failure
        """
        self.upload_attempts += 1
        self.last_upload_error = error_message
        self.uploaded_at = None

    @property
    def is_uploaded(self) -> bool:
        """Check if this metrics record has been successfully uploaded."""
        return self.uploaded_at is not None

    @property
    def needs_retry(self) -> bool:
        """Check if this metrics record needs upload retry (failed but not too many attempts)."""
        return not self.is_uploaded and self.upload_attempts < 3

    def __repr__(self) -> str:
        return (
            f"<TelemetryMetrics(id='{self.id}', "
            f"collected_at='{self.collected_at}', "
            f'uploaded={self.is_uploaded})>'
        )

    class Config:
        from_attributes = True
