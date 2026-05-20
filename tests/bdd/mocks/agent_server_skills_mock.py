"""Mock agent-server /api/skills endpoint for BDD tests.

Simulates the agent-server's skill loading API without requiring a real
agent-server deployment. Tracks requests and returns configured skill sets
based on load flags (load_public, load_user, load_project, load_org).

Usage:
    skills_api = MockAgentServerSkillsAPI()
    skills_api.set_project_skills([
        {'name': 'create_file', 'content': '...', 'triggers': ['file']}
    ])
    response = await skills_api.handle_request({
        'load_public': False,
        'load_project': True,
        'project_dir': '/workspace'
    })
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockAgentServerSkillsAPI:
    """Mock implementation of agent-server /api/skills endpoint.

    Attributes:
        project_skills: List of project-specific skill definitions
        global_skills: List of public/global skill definitions
        user_skills: List of user personal skill definitions
        org_skills: List of organization skill definitions
        call_history: Record of all requests received
        should_fail: Whether to simulate failure
        failure_status: HTTP status code for simulated failure
        failure_message: Error message for simulated failure
    """

    project_skills: list[dict[str, Any]] = field(default_factory=list)
    global_skills: list[dict[str, Any]] = field(default_factory=list)
    user_skills: list[dict[str, Any]] = field(default_factory=list)
    org_skills: list[dict[str, Any]] = field(default_factory=list)
    call_history: list[dict[str, Any]] = field(default_factory=list)
    should_fail: bool = False
    failure_status: int = 500
    failure_message: str = 'Internal Server Error'
    malformed_response: bool = False

    async def handle_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle POST /api/skills request from app-server.

        Args:
            payload: Request body containing:
                - load_public: bool - whether to load global skills
                - load_user: bool - whether to load user skills
                - load_project: bool - whether to load project skills
                - load_org: bool - whether to load org skills
                - project_dir: str - path to project directory
                - org_config: dict - org configuration (optional)
                - sandbox_config: dict - sandbox configuration (optional)

        Returns:
            dict with 'skills' and 'sources' keys

        Raises:
            Exception: If should_fail is True
        """
        # Record the request
        self.call_history.append(payload)

        # Simulate failures if configured
        if self.should_fail:
            raise Exception(f'[{self.failure_status}] {self.failure_message}')

        # Return malformed response if configured
        if self.malformed_response:
            return {
                'invalid_key': 'this_response_is_malformed',
                # Missing required 'skills' and 'sources' keys
            }

        # Build response based on load flags
        skills: list[dict[str, Any]] = []
        sources: dict[str, int] = {}

        if payload.get('load_public', False):
            skills.extend(self.global_skills)
            sources['public'] = len(self.global_skills)

        if payload.get('load_user', False):
            skills.extend(self.user_skills)
            sources['user'] = len(self.user_skills)

        if payload.get('load_project', False):
            skills.extend(self.project_skills)
            sources['project'] = len(self.project_skills)

        if payload.get('load_org', False):
            skills.extend(self.org_skills)
            sources['org'] = len(self.org_skills)

        return {
            'skills': skills,
            'sources': sources,
        }

    def handle_request_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle POST /api/skills request synchronously (for BDD tests).

        This is a synchronous version of handle_request for use in synchronous test steps.

        Args:
            payload: Request body containing load flags and config

        Returns:
            dict with 'skills' and 'sources' keys

        Raises:
            Exception: If should_fail is True
        """
        # Record the request
        self.call_history.append(payload)

        # Simulate failures if configured
        if self.should_fail:
            raise Exception(f'[{self.failure_status}] {self.failure_message}')

        # Return malformed response if configured
        if self.malformed_response:
            return {
                'invalid_key': 'this_response_is_malformed',
                # Missing required 'skills' and 'sources' keys
            }

        # Build response based on load flags
        skills: list[dict[str, Any]] = []
        sources: dict[str, int] = {}

        if payload.get('load_public', False):
            skills.extend(self.global_skills)
            sources['public'] = len(self.global_skills)

        if payload.get('load_user', False):
            skills.extend(self.user_skills)
            sources['user'] = len(self.user_skills)

        if payload.get('load_project', False):
            skills.extend(self.project_skills)
            sources['project'] = len(self.project_skills)

        if payload.get('load_org', False):
            skills.extend(self.org_skills)
            sources['org'] = len(self.org_skills)

        return {
            'skills': skills,
            'sources': sources,
        }

    def set_project_skills(self, skills: list[dict[str, Any]] | list[str]) -> None:
        """Configure project skills.

        Args:
            skills: List of skill dicts or skill names.
                    If names, creates minimal dicts with name and empty content.
        """
        if not skills:
            self.project_skills = []
            return

        # Convert names to dicts if needed
        if isinstance(skills[0], str):
            self.project_skills = [
                {
                    'name': name,
                    'content': f'# Skill: {name}\ndef {name}(): pass',
                    'triggers': [name],
                    'source': 'project',
                    'is_agentskills_format': False,
                }
                for name in skills
            ]
        else:
            self.project_skills = skills

    def set_global_skills(self, skills: list[dict[str, Any]] | list[str]) -> None:
        """Configure global/public skills.

        Args:
            skills: List of skill dicts or skill names.
        """
        if not skills:
            self.global_skills = []
            return

        if isinstance(skills[0], str):
            self.global_skills = [
                {
                    'name': name,
                    'content': f'# Skill: {name}\ndef {name}(): pass',
                    'triggers': [name],
                    'source': 'public',
                    'is_agentskills_format': False,
                }
                for name in skills
            ]
        else:
            self.global_skills = skills

    def set_user_skills(self, skills: list[dict[str, Any]] | list[str]) -> None:
        """Configure user personal skills.

        Args:
            skills: List of skill dicts or skill names.
        """
        if not skills:
            self.user_skills = []
            return

        if isinstance(skills[0], str):
            self.user_skills = [
                {
                    'name': name,
                    'content': f'# Skill: {name}\ndef {name}(): pass',
                    'triggers': [name],
                    'source': 'user',
                    'is_agentskills_format': False,
                }
                for name in skills
            ]
        else:
            self.user_skills = skills

    def set_org_skills(self, skills: list[dict[str, Any]] | list[str]) -> None:
        """Configure organization skills.

        Args:
            skills: List of skill dicts or skill names.
        """
        if not skills:
            self.org_skills = []
            return

        if isinstance(skills[0], str):
            self.org_skills = [
                {
                    'name': name,
                    'content': f'# Skill: {name}\ndef {name}(): pass',
                    'triggers': [name],
                    'source': 'org',
                    'is_agentskills_format': False,
                }
                for name in skills
            ]
        else:
            self.org_skills = skills

    def assert_called_with(
        self,
        load_public: bool | None = None,
        load_user: bool | None = None,
        load_project: bool | None = None,
        load_org: bool | None = None,
    ) -> bool:
        """Assert API was called with specific parameters.

        Args:
            load_public: Expected value for load_public flag
            load_user: Expected value for load_user flag
            load_project: Expected value for load_project flag
            load_org: Expected value for load_org flag

        Returns:
            True if any call matched all provided conditions

        Raises:
            AssertionError: If no call matches conditions
        """
        for call in self.call_history:
            if load_public is not None and call.get('load_public') != load_public:
                continue
            if load_user is not None and call.get('load_user') != load_user:
                continue
            if load_project is not None and call.get('load_project') != load_project:
                continue
            if load_org is not None and call.get('load_org') != load_org:
                continue
            # All conditions matched
            return True

        # No matching call found
        conditions = []
        if load_public is not None:
            conditions.append(f'load_public={load_public}')
        if load_user is not None:
            conditions.append(f'load_user={load_user}')
        if load_project is not None:
            conditions.append(f'load_project={load_project}')
        if load_org is not None:
            conditions.append(f'load_org={load_org}')

        raise AssertionError(
            f'Expected call with {", ".join(conditions)} but got {len(self.call_history)} calls: '
            f'{self.call_history}'
        )

    def get_call_count(self) -> int:
        """Get number of times endpoint was called.

        Returns:
            Number of requests received
        """
        return len(self.call_history)

    def get_last_call(self) -> dict[str, Any] | None:
        """Get the last request received.

        Returns:
            Last request dict, or None if no calls recorded yet
        """
        if not self.call_history:
            return None
        return self.call_history[-1]

    def set_large_skills_response(self, count: int = 1000) -> None:
        """Generate a large number of project skills for performance testing.

        Args:
            count: Number of skills to generate
        """
        self.project_skills = [
            {
                'name': f'skill_{i}',
                'content': f'# Skill {i}\ndef skill_{i}(): pass',
                'triggers': [f'skill_{i}'],
                'source': 'project',
                'is_agentskills_format': False,
            }
            for i in range(count)
        ]

    def set_skills_with_mixed_validity(self) -> None:
        """Set project skills with a mix of valid and invalid structures.

        Valid skills have 'name' field; invalid ones are missing it or have
        unexpected data types. Used to test graceful handling of malformed skills.
        """
        self.project_skills = [
            {
                'name': 'valid_skill_1',
                'content': '# Valid skill',
                'triggers': ['valid_1'],
                'source': 'project',
            },
            {
                # Missing 'name' field
                'content': 'Invalid: no name',
                'triggers': ['invalid'],
            },
            {
                'name': 'valid_skill_2',
                'content': '# Another valid skill',
                'triggers': ['valid_2'],
            },
            {
                'name': 123,  # Invalid: name is not a string
                'content': 'Invalid: name is int',
                'triggers': ['invalid_2'],
            },
            {
                'name': 'valid_skill_3',
                'content': '# Third valid skill',
                'triggers': ['valid_3'],
            },
        ]

    def set_empty_skills_response(self) -> None:
        """Clear all skill lists to simulate empty response."""
        self.project_skills = []
        self.global_skills = []
        self.user_skills = []
        self.org_skills = []

    def reset(self) -> None:
        """Reset all state for next test."""
        self.call_history.clear()
        self.should_fail = False
        self.malformed_response = False

    def set_failure(
        self, status: int = 500, message: str = 'Internal Server Error'
    ) -> None:
        """Configure endpoint to fail.

        Args:
            status: HTTP status code to simulate
            message: Error message
        """
        self.should_fail = True
        self.failure_status = status
        self.failure_message = message

    def set_malformed_response(self, malformed: bool = True) -> None:
        """Configure endpoint to return malformed response.

        Args:
            malformed: Whether to return invalid JSON structure
        """
        self.malformed_response = malformed

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'MockAgentServerSkillsAPI('
            f'calls={len(self.call_history)}, '
            f'project={len(self.project_skills)}, '
            f'global={len(self.global_skills)}, '
            f'user={len(self.user_skills)}, '
            f'org={len(self.org_skills)})'
        )
