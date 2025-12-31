from __future__ import annotations

import pathlib
from collections.abc import Mapping

from pydantic import BaseModel, Field, field_validator, model_validator

from openhands.sdk.context.prompts import render_template
from openhands.sdk.context.skills import (
    Skill,
    SkillKnowledge,
    load_public_skills,
    load_user_skills,
)
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.utils.model_prompt_spec import get_model_prompt_spec
from openhands.sdk.logger import get_logger
from openhands.sdk.secret import SecretSource, SecretValue


logger = get_logger(__name__)

PROMPT_DIR = pathlib.Path(__file__).parent / "prompts" / "templates"


class AgentContext(BaseModel):
    """Central structure for managing prompt extension.

    AgentContext unifies all the contextual inputs that shape how the system
    extends and interprets user prompts. It combines both static environment
    details and dynamic, user-activated extensions from skills.

    Specifically, it provides:
    - **Repository context / Repo Skills**: Information about the active codebase,
      branches, and repo-specific instructions contributed by repo skills.
    - **Runtime context**: Current execution environment (hosts, working
      directory, secrets, date, etc.).
    - **Conversation instructions**: Optional task- or channel-specific rules
      that constrain or guide the agent’s behavior across the session.
    - **Knowledge Skills**: Extensible components that can be triggered by user input
      to inject knowledge or domain-specific guidance.

    Together, these elements make AgentContext the primary container responsible
    for assembling, formatting, and injecting all prompt-relevant context into
    LLM interactions.
    """  # noqa: E501

    skills: list[Skill] = Field(
        default_factory=list,
        description="List of available skills that can extend the user's input.",
    )
    system_message_suffix: str | None = Field(
        default=None, description="Optional suffix to append to the system prompt."
    )
    user_message_suffix: str | None = Field(
        default=None, description="Optional suffix to append to the user's message."
    )
    load_user_skills: bool = Field(
        default=False,
        description=(
            "Whether to automatically load user skills from ~/.openhands/skills/ "
            "and ~/.openhands/microagents/ (for backward compatibility). "
        ),
    )
    load_public_skills: bool = Field(
        default=False,
        description=(
            "Whether to automatically load skills from the public OpenHands "
            "skills repository at https://github.com/OpenHands/skills. "
            "This allows you to get the latest skills without SDK updates."
        ),
    )
    secrets: Mapping[str, SecretValue] | None = Field(
        default=None,
        description=(
            "Dictionary mapping secret keys to values or secret sources. "
            "Secrets are used for authentication and sensitive data handling. "
            "Values can be either strings or SecretSource instances "
            "(str | SecretSource)."
        ),
    )

    @field_validator("skills")
    @classmethod
    def _validate_skills(cls, v: list[Skill], _info):
        if not v:
            return v
        # Check for duplicate skill names
        seen_names = set()
        for skill in v:
            if skill.name in seen_names:
                raise ValueError(f"Duplicate skill name found: {skill.name}")
            seen_names.add(skill.name)
        return v

    @model_validator(mode="after")
    def _load_user_skills(self):
        """Load user skills from home directory if enabled."""
        if not self.load_user_skills:
            return self

        try:
            user_skills = load_user_skills()
            # Merge user skills with explicit skills, avoiding duplicates
            existing_names = {skill.name for skill in self.skills}
            for user_skill in user_skills:
                if user_skill.name not in existing_names:
                    self.skills.append(user_skill)
                else:
                    logger.warning(
                        f"Skipping user skill '{user_skill.name}' "
                        f"(already in explicit skills)"
                    )
        except Exception as e:
            logger.warning(f"Failed to load user skills: {str(e)}")

        return self

    @model_validator(mode="after")
    def _load_public_skills(self):
        """Load public skills from OpenHands skills repository if enabled."""
        if not self.load_public_skills:
            return self
        try:
            public_skills = load_public_skills()
            # Merge public skills with explicit skills, avoiding duplicates
            existing_names = {skill.name for skill in self.skills}
            for public_skill in public_skills:
                if public_skill.name not in existing_names:
                    self.skills.append(public_skill)
                else:
                    logger.warning(
                        f"Skipping public skill '{public_skill.name}' "
                        f"(already in existing skills)"
                    )
        except Exception as e:
            logger.warning(f"Failed to load public skills: {str(e)}")
        return self

    def get_secret_infos(self) -> list[dict[str, str]]:
        """Get secret information (name and description) from the secrets field.

        Returns:
            List of dictionaries with 'name' and 'description' keys.
            Returns an empty list if no secrets are configured.
            Description will be None if not available.
        """
        if not self.secrets:
            return []
        secret_infos = []
        for name, secret_value in self.secrets.items():
            description = None
            if isinstance(secret_value, SecretSource):
                description = secret_value.description
            secret_infos.append({"name": name, "description": description})
        return secret_infos

    def get_system_message_suffix(
        self,
        llm_model: str | None = None,
        llm_model_canonical: str | None = None,
    ) -> str | None:
        """Get the system message with repo skill content and custom suffix.

        Custom suffix can typically includes:
        - Repository information (repo name, branch name, PR number, etc.)
        - Runtime information (e.g., available hosts, current date)
        - Conversation instructions (e.g., user preferences, task details)
        - Repository-specific instructions (collected from repo skills)
        """
        repo_skills = [s for s in self.skills if s.trigger is None]

        # Gate vendor-specific repo skills based on model family.
        if llm_model or llm_model_canonical:
            spec = get_model_prompt_spec(llm_model or "", llm_model_canonical)
            family = (spec.family or "").lower()
            if family:
                filtered: list[Skill] = []
                for s in repo_skills:
                    n = (s.name or "").lower()
                    if n == "claude" and not (
                        "anthropic" in family or "claude" in family
                    ):
                        continue
                    if n == "gemini" and not (
                        "gemini" in family or "google_gemini" in family
                    ):
                        continue
                    filtered.append(s)
                repo_skills = filtered

        logger.debug(f"Triggered {len(repo_skills)} repository skills: {repo_skills}")
        # Build the workspace context information
        secret_infos = self.get_secret_infos()
        if repo_skills or self.system_message_suffix or secret_infos:
            formatted_text = render_template(
                prompt_dir=str(PROMPT_DIR),
                template_name="system_message_suffix.j2",
                repo_skills=repo_skills,
                system_message_suffix=self.system_message_suffix or "",
                secret_infos=secret_infos,
            ).strip()
            return formatted_text
        elif self.system_message_suffix and self.system_message_suffix.strip():
            return self.system_message_suffix.strip()
        return None

    def get_user_message_suffix(
        self, user_message: Message, skip_skill_names: list[str]
    ) -> tuple[TextContent, list[str]] | None:
        """Augment the user’s message with knowledge recalled from skills.

        This works by:
        - Extracting the text content of the user message
        - Matching skill triggers against the query
        - Returning formatted knowledge and triggered skill names if relevant skills were triggered
        """  # noqa: E501

        user_message_suffix = None
        if self.user_message_suffix and self.user_message_suffix.strip():
            user_message_suffix = self.user_message_suffix.strip()

        query = "\n".join(
            c.text for c in user_message.content if isinstance(c, TextContent)
        ).strip()
        recalled_knowledge: list[SkillKnowledge] = []
        # skip empty queries, but still return user_message_suffix if it exists
        if not query:
            if user_message_suffix:
                return TextContent(text=user_message_suffix), []
            return None
        # Search for skill triggers in the query
        for skill in self.skills:
            if not isinstance(skill, Skill):
                continue
            trigger = skill.match_trigger(query)
            if trigger and skill.name not in skip_skill_names:
                logger.info(
                    "Skill '%s' triggered by keyword '%s'",
                    skill.name,
                    trigger,
                )
                recalled_knowledge.append(
                    SkillKnowledge(
                        name=skill.name,
                        trigger=trigger,
                        content=skill.content,
                    )
                )
        if recalled_knowledge:
            formatted_skill_text = render_template(
                prompt_dir=str(PROMPT_DIR),
                template_name="skill_knowledge_info.j2",
                triggered_agents=recalled_knowledge,
            )
            if user_message_suffix:
                formatted_skill_text += "\n" + user_message_suffix
            return TextContent(text=formatted_skill_text), [
                k.name for k in recalled_knowledge
            ]

        if user_message_suffix:
            return TextContent(text=user_message_suffix), []
        return None
