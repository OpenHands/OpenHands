from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints


class VerifiedModelCreate(BaseModel):
    model_name: Annotated[
        str,
        StringConstraints(max_length=255),
    ]
    provider: Annotated[
        str,
        StringConstraints(max_length=100),
    ]
    is_enabled: bool = True


class VerifiedModel(VerifiedModelCreate):
    id: int
    created_at: datetime
    updated_at: datetime


class VerifiedModelUpdate(BaseModel):
    is_enabled: bool | None = None


class VerifiedModelPage(BaseModel):
    """Paginated response model for verified model list."""

    items: list[VerifiedModel]
    next_page_id: str | None = None
