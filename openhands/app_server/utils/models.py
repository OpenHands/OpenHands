from pydantic import BaseModel


class EditResponse(BaseModel):
    """General response to an edit operation"""

    message: str
