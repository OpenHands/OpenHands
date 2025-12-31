from datetime import UTC, datetime

from pydantic import BaseModel, Field


class InputMetadata(BaseModel):
    """Metadata for task skill inputs."""

    name: str = Field(description="Name of the input parameter")
    description: str = Field(description="Description of the input parameter")


class SkillKnowledge(BaseModel):
    """Represents knowledge from a triggered skill."""

    name: str = Field(description="The name of the skill that was triggered")
    trigger: str = Field(description="The word that triggered this skill")
    content: str = Field(description="The actual content/knowledge from the skill")


class SkillResponse(BaseModel):
    """Response model for skills endpoint.

    Note: This model only includes basic metadata that can be determined
    without parsing skill content. Use the separate content API
    to get detailed skill information.
    """

    name: str = Field(description="The name of the skill")
    path: str = Field(description="The path or identifier of the skill")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the skill was created",
    )


class SkillContentResponse(BaseModel):
    """Response model for individual skill content endpoint."""

    content: str = Field(description="The full content of the skill")
    path: str = Field(description="The path or identifier of the skill")
    triggers: list[str] = Field(
        description="List of triggers associated with the skill"
    )
    git_provider: str | None = Field(
        None,
        description="Git provider if the skill is sourced from a Git repository",
    )
