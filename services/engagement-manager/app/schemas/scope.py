from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RuleType = Literal["allow", "deny"]
TargetType = Literal["ip", "cidr", "domain", "url"]


class ScopeRuleCreate(BaseModel):
    rule_type: RuleType
    target_type: TargetType
    target_value: str
    note: str | None = None


class ScopeRuleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    engagement_id: uuid.UUID
    rule_type: str
    target_type: str
    target_value: str
    note: str | None
    created_at: datetime


class AuthorizeScopeRequest(BaseModel):
    scope_document_url: str
    scope_rules: list[ScopeRuleCreate] = Field(default_factory=list)
