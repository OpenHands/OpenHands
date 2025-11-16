"""Minimal StoredConversationMetadata for enterprise tests.

This module provides a minimal StoredConversationMetadata class that avoids
the broken SDK import chain in the main codebase, allowing enterprise tests
to run successfully.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from storage.base import Base


class StoredConversationMetadata(Base):
    """Minimal conversation metadata model for enterprise tests."""

    __tablename__ = 'conversation_metadata'
    __table_args__ = {'extend_existing': True}

    conversation_id = Column(String, primary_key=True)
    github_user_id = Column(String, nullable=True)
    user_id = Column(String, nullable=False)
    selected_repository = Column(String, nullable=True)
    selected_branch = Column(String, nullable=True)
    git_provider = Column(String, nullable=True)
    title = Column(String, nullable=True)
    last_updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    trigger = Column(String, nullable=True)
    pr_number = Column(JSON, nullable=True)

    # Cost and token metrics
    accumulated_cost = Column(Float, default=0.0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    max_budget_per_task = Column(Float, nullable=True)
    cache_read_tokens = Column(Integer, default=0)
    cache_write_tokens = Column(Integer, default=0)
    reasoning_tokens = Column(Integer, default=0)
    context_window = Column(Integer, default=0)
    per_turn_token = Column(Integer, default=0)

    # LLM model used for the conversation
    llm_model = Column(String, nullable=True)

    conversation_version = Column(String, nullable=False, default='V0')
    sandbox_id = Column(String, nullable=True)

    def __init__(
        self,
        conversation_id: str,
        user_id: str,
        github_user_id: Optional[str] = None,
        selected_repository: Optional[str] = None,
        selected_branch: Optional[str] = None,
        git_provider: Optional[str] = None,
        title: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_updated_at: Optional[datetime] = None,
        trigger: Optional[str] = None,
        pr_number: Optional[list] = None,
        accumulated_cost: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        max_budget_per_task: Optional[float] = None,
        cache_read_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
        per_turn_token: Optional[int] = None,
        llm_model: Optional[str] = None,
        conversation_version: str = 'V0',
        sandbox_id: Optional[str] = None,
    ):
        self.conversation_id = conversation_id
        self.github_user_id = github_user_id
        self.user_id = user_id
        self.selected_repository = selected_repository
        self.selected_branch = selected_branch
        self.git_provider = git_provider
        self.title = title
        self.created_at = created_at or datetime.utcnow()
        self.last_updated_at = last_updated_at or datetime.utcnow()
        self.trigger = trigger
        self.pr_number = pr_number
        self.accumulated_cost = accumulated_cost or 0.0
        self.prompt_tokens = prompt_tokens or 0
        self.completion_tokens = completion_tokens or 0
        self.total_tokens = total_tokens or 0
        self.max_budget_per_task = max_budget_per_task
        self.cache_read_tokens = cache_read_tokens or 0
        self.cache_write_tokens = cache_write_tokens or 0
        self.reasoning_tokens = reasoning_tokens or 0
        self.context_window = context_window or 0
        self.per_turn_token = per_turn_token or 0
        self.llm_model = llm_model
        self.conversation_version = conversation_version
        self.sandbox_id = sandbox_id


__all__ = ['StoredConversationMetadata']
