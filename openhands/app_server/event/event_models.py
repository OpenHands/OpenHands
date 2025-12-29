from enum import Enum
from pydantic import BaseModel
from typing import Any, List, Optional

class EventSortOrder(str, Enum):
    TIMESTAMP = 'TIMESTAMP'
    TIMESTAMP_DESC = 'TIMESTAMP_DESC'

class EventPage(BaseModel):
    items: List[Any]
    next_page_id: Optional[str] = None
