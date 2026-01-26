"""Tests for the microagent system."""

import tempfile
from pathlib import Path

import pytest

from openhands.microagent import (
    BaseMicroagent,
    KnowledgeMicroagent,
    MicroagentMetadata,
    MicroagentType,
    RepoMicroagent,
    load_microagents_from_dir,
)

CONTENT = '# dummy header\ndummy content\n## dummy subheader\ndummy subcontent\n'


def test_legacy_micro_agent_load(tmp_path):
    """Test loading of legacy microagents."""
    legacy_file = tmp_path / '.openhands_instructions'
    legacy_file.write_text(CONTENT)

    # Pass microagent_dir (tmp_path in this case) to load
    micro_agent = BaseMicroagent.load(legacy_file, tmp_path)
    assert isinstance(micro_agent, RepoMicroagent)
    assert micro_agent.name == 'repo_legacy'  # Legacy name is hardcoded
    assert micro_agent.content == CONTENT
    assert micro_agent.type == MicroagentType.REPO_KNOWLEDGE


@pytest.fixture
def temp_microagents_dir():
    """Create a temporary directory with test microagents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create test knowledge agent (type inferred from triggers)
        knowledge_agent = """---
# type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - test
  - pytest
---

# Test Guidelines

Testing best practices and guidelines.
"""
        (root / 'knowledge.md').write_text(knowledge_agent)

        # Create test repo agent (type inferred from lack of triggers)
        repo_agent = """---
# type: repo
version: 1.0.0
agent: CodeActAgent
---

# Test Repository Agent

Repository-specific test instructions.
"""
        (root / 'repo.md').write_text(repo_agent)

        yield root


def test_knowledge_agent():
    """Test knowledge agent functionality."""
    # Note: We still pass type to the constructor here, as it expects it.
    # The loader infers the type before calling the constructor.
    agent = KnowledgeMicroagent(
        name='test',
        content='Test content',
        metadata=MicroagentMetadata(name='test', triggers=['test', 'pytest']),
        source='test.md',
        type=MicroagentType.KNOWLEDGE,  # Constructor still needs type
    )

    assert agent.match_trigger('running a test') == 'test'
    assert agent.match_trigger('using pytest') == 'test'
    assert agent.match_trigger('no match here') is None
    assert agent.triggers == ['test', 'pytest']


def test_load_microagents(temp_microagents_dir):
    """Test loading microagents from directory."""
    repo_agents, knowledge_agents = load_microagents_from_dir(temp_microagents_dir)

    # Check knowledge agents (name derived from filename: knowledge.md -> 'knowledge')
    assert len(knowledge_agents) == 1
    agent_k = knowledge_agents['knowledge']
    assert isinstance(agent_k, KnowledgeMicroagent)
    assert agent_k.type == MicroagentType.KNOWLEDGE  # Check inferred type
    assert 'test' in agent_k.triggers

    # Check repo agents (name derived from filename: repo.md -> 'repo')
    assert len(repo_agents) == 1
    agent_r = repo_agents['repo']
    assert isinstance(agent_r, RepoMicroagent)
    assert agent_r.type == MicroagentType.REPO_KNOWLEDGE  # Check inferred type


def test_load_microagents_with_nested_dirs(temp_microagents_dir):
    """Test loading microagents from nested directories."""
    # Create nested knowledge agent
    nested_dir = temp_microagents_dir / 'nested' / 'dir'
    nested_dir.mkdir(parents=True)
    nested_agent = """---
# type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - nested
---

# Nested Test Guidelines

Testing nested directory loading.
"""
    (nested_dir / 'nested.md').write_text(nested_agent)

    repo_agents, knowledge_agents = load_microagents_from_dir(temp_microagents_dir)

    # Check that we can find the nested agent (name derived from path: nested/dir/nested.md -> 'nested/dir/nested')
    assert (
        len(knowledge_agents) == 2
    )  # Original ('knowledge') + nested ('nested/dir/nested')
    agent_n = knowledge_agents['nested/dir/nested']
    assert isinstance(agent_n, KnowledgeMicroagent)
    assert agent_n.type == MicroagentType.KNOWLEDGE  # Check inferred type
    assert 'nested' in agent_n.triggers


def test_load_microagents_with_trailing_slashes(temp_microagents_dir):
    """Test loading microagents when directory paths have trailing slashes."""
    # Create a directory with trailing slash
    knowledge_dir = temp_microagents_dir / 'test_knowledge/'
    knowledge_dir.mkdir(exist_ok=True)
    knowledge_agent = """---
# type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - trailing
---

# Trailing Slash Test

Testing loading with trailing slashes.
"""
    (knowledge_dir / 'trailing.md').write_text(knowledge_agent)

    repo_agents, knowledge_agents = load_microagents_from_dir(
        str(temp_microagents_dir) + '/'  # Add trailing slash to test
    )

    # Check that we can find the agent despite trailing slashes (name derived from path: test_knowledge/trailing.md -> 'test_knowledge/trailing')
    assert (
        len(knowledge_agents) == 2
    )  # Original ('knowledge') + trailing ('test_knowledge/trailing')
    agent_t = knowledge_agents['test_knowledge/trailing']
    assert isinstance(agent_t, KnowledgeMicroagent)
    assert agent_t.type == MicroagentType.KNOWLEDGE  # Check inferred type
    assert 'trailing' in agent_t.triggers


def test_invalid_microagent_type(temp_microagents_dir):
    """Test loading a microagent with an invalid type."""
    # Create a microagent with an invalid type
    invalid_agent = """---
name: invalid_type_agent
type: invalid_type
version: 1.0.0
agent: CodeActAgent
triggers:
  - test
---

# Invalid Type Test

This microagent has an invalid type.
"""
    invalid_file = temp_microagents_dir / 'invalid_type.md'
    invalid_file.write_text(invalid_agent)

    # Attempt to load the microagent should raise a MicroagentValidationError
    from openhands.core.exceptions import MicroagentValidationError

    with pytest.raises(MicroagentValidationError) as excinfo:
        load_microagents_from_dir(temp_microagents_dir)

    # Check that the error message contains helpful information
    error_msg = str(excinfo.value)
    assert 'invalid_type.md' in error_msg
    assert 'Invalid "type" value: "invalid_type"' in error_msg
    assert 'Valid types are:' in error_msg
    assert '"knowledge"' in error_msg
    assert '"repo"' in error_msg
    assert '"task"' in error_msg


def test_cursorrules_file_load():
    """Test loading .cursorrules file as a RepoMicroagent."""
    cursorrules_content = """Always use Python for new files.
Follow the existing code style.
Add proper error handling."""

    cursorrules_path = Path('.cursorrules')

    # Test loading .cursorrules file directly
    agent = BaseMicroagent.load(cursorrules_path, file_content=cursorrules_content)

    # Verify it's loaded as a RepoMicroagent
    assert isinstance(agent, RepoMicroagent)
    assert agent.name == 'cursorrules'
    assert agent.content == cursorrules_content
    assert agent.type == MicroagentType.REPO_KNOWLEDGE
    assert agent.metadata.name == 'cursorrules'
    assert agent.source == str(cursorrules_path)


def test_microagent_version_as_integer():
    """Test loading a microagent with version as integer (reproduces the bug)."""
    # Create a microagent with version as an unquoted integer
    # This should be parsed as an integer by YAML but converted to string by our code
    microagent_content = """---
name: test_agent
type: knowledge
version: 2512312
agent: CodeActAgent
triggers:
  - test
---

# Test Agent

This is a test agent with integer version.
"""

    test_path = Path('test_agent.md')

    # This should not raise an error even though version is an integer in YAML
    agent = BaseMicroagent.load(test_path, file_content=microagent_content)

    # Verify the agent was loaded correctly
    assert isinstance(agent, KnowledgeMicroagent)
    assert agent.name == 'test_agent'
    assert agent.metadata.version == '2512312'  # Should be converted to string
    assert isinstance(agent.metadata.version, str)  # Ensure it's actually a string
    assert agent.type == MicroagentType.KNOWLEDGE


def test_microagent_version_as_float():
    """Test loading a microagent with version as float."""
    # Create a microagent with version as an unquoted float
    microagent_content = """---
name: test_agent_float
type: knowledge
version: 1.5
agent: CodeActAgent
triggers:
  - test
---

# Test Agent Float

This is a test agent with float version.
"""

    test_path = Path('test_agent_float.md')

    # This should not raise an error even though version is a float in YAML
    agent = BaseMicroagent.load(test_path, file_content=microagent_content)

    # Verify the agent was loaded correctly
    assert isinstance(agent, KnowledgeMicroagent)
    assert agent.name == 'test_agent_float'
    assert agent.metadata.version == '1.5'  # Should be converted to string
    assert isinstance(agent.metadata.version, str)  # Ensure it's actually a string
    assert agent.type == MicroagentType.KNOWLEDGE


def test_microagent_version_as_string_unchanged():
    """Test loading a microagent with version as string (should remain unchanged)."""
    # Create a microagent with version as a quoted string
    microagent_content = """---
name: test_agent_string
type: knowledge
version: "1.0.0"
agent: CodeActAgent
triggers:
  - test
---

# Test Agent String

This is a test agent with string version.
"""

    test_path = Path('test_agent_string.md')

    # This should work normally
    agent = BaseMicroagent.load(test_path, file_content=microagent_content)

    # Verify the agent was loaded correctly
    assert isinstance(agent, KnowledgeMicroagent)
    assert agent.name == 'test_agent_string'
    assert agent.metadata.version == '1.0.0'  # Should remain as string
    assert isinstance(agent.metadata.version, str)  # Ensure it's actually a string
    assert agent.type == MicroagentType.KNOWLEDGE


@pytest.fixture
def temp_microagents_dir_with_cursorrules():
    """Create a temporary directory with test microagents and .cursorrules file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create .openhands/microagents directory structure
        microagents_dir = root / '.openhands' / 'microagents'
        microagents_dir.mkdir(parents=True, exist_ok=True)

        # Create .cursorrules file in repository root
        cursorrules_content = """Always use TypeScript for new files.
Follow the existing code style."""
        (root / '.cursorrules').write_text(cursorrules_content)

        # Create test repo agent
        repo_agent = """---
# type: repo
version: 1.0.0
agent: CodeActAgent
---

# Test Repository Agent

Repository-specific test instructions.
"""
        (microagents_dir / 'repo.md').write_text(repo_agent)

        yield root


def test_load_microagents_with_cursorrules(temp_microagents_dir_with_cursorrules):
    """Test loading microagents when .cursorrules file exists."""
    microagents_dir = (
        temp_microagents_dir_with_cursorrules / '.openhands' / 'microagents'
    )

    repo_agents, knowledge_agents = load_microagents_from_dir(microagents_dir)

    # Verify that .cursorrules file was loaded as a RepoMicroagent
    assert len(repo_agents) == 2  # repo.md + .cursorrules
    assert 'repo' in repo_agents
    assert 'cursorrules' in repo_agents

    # Check .cursorrules agent
    cursorrules_agent = repo_agents['cursorrules']
    assert isinstance(cursorrules_agent, RepoMicroagent)
    assert cursorrules_agent.name == 'cursorrules'
    assert 'Always use TypeScript for new files' in cursorrules_agent.content
    assert cursorrules_agent.type == MicroagentType.REPO_KNOWLEDGE


@pytest.fixture
def temp_dir_with_cursorrules_only():
    """Create a temporary directory with only .cursorrules file (no .openhands/microagents directory)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create .cursorrules file in repository root
        cursorrules_content = """Always use Python for new files.
Follow PEP 8 style guidelines."""
        (root / '.cursorrules').write_text(cursorrules_content)

        # Note: We intentionally do NOT create .openhands/microagents directory
        yield root


def test_load_cursorrules_without_microagents_dir(temp_dir_with_cursorrules_only):
    """Test loading .cursorrules file when .openhands/microagents directory doesn't exist.

    This test reproduces the bug where .cursorrules is only loaded when
    .openhands/microagents directory exists.
    """
    # Try to load from non-existent microagents directory
    microagents_dir = temp_dir_with_cursorrules_only / '.openhands' / 'microagents'

    repo_agents, knowledge_agents = load_microagents_from_dir(microagents_dir)

    # This should find the .cursorrules file even though microagents_dir doesn't exist
    assert len(repo_agents) == 1  # Only .cursorrules
    assert 'cursorrules' in repo_agents
    assert len(knowledge_agents) == 0

    # Check .cursorrules agent
    cursorrules_agent = repo_agents['cursorrules']
    assert isinstance(cursorrules_agent, RepoMicroagent)
    assert cursorrules_agent.name == 'cursorrules'
    assert 'Always use Python for new files' in cursorrules_agent.content
    assert cursorrules_agent.type == MicroagentType.REPO_KNOWLEDGE


def test_agents_md_file_load():
    """Test loading AGENTS.md file as a RepoMicroagent."""
    agents_content = """# Project Setup

## Setup commands

- Install deps: `npm install`
- Start dev server: `npm run dev`
- Run tests: `npm test`

## Code style

- TypeScript strict mode
- Single quotes, no semicolons
- Use functional patterns where possible"""

    agents_path = Path('AGENTS.md')

    # Test loading AGENTS.md file directly
    agent = BaseMicroagent.load(agents_path, file_content=agents_content)

    # Verify it's loaded as a RepoMicroagent
    assert isinstance(agent, RepoMicroagent)
    assert agent.name == 'agents'
    assert agent.content == agents_content
    assert agent.type == MicroagentType.REPO_KNOWLEDGE
    assert agent.metadata.name == 'agents'
    assert agent.source == str(agents_path)


def test_agents_md_case_insensitive():
    """Test that AGENTS.md loading is case-insensitive."""
    agents_content = """# Development Guide

Use TypeScript for all new files."""

    test_cases = ['AGENTS.md', 'agents.md', 'AGENT.md', 'agent.md']

    for filename in test_cases:
        agents_path = Path(filename)
        agent = BaseMicroagent.load(agents_path, file_content=agents_content)

        assert isinstance(agent, RepoMicroagent)
        assert agent.name == 'agents'
        assert agent.content == agents_content
        assert agent.type == MicroagentType.REPO_KNOWLEDGE


@pytest.fixture
def temp_dir_with_agents_md_only():
    """Create a temporary directory with only AGENTS.md file (no .openhands/microagents directory)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create AGENTS.md file in repository root
        agents_content = """# Development Guide

## Setup commands

- Install deps: `poetry install`
- Start dev server: `poetry run python app.py`
- Run tests: `poetry run pytest`

## Code style

- Python 3.12+
- Follow PEP 8 guidelines
- Use type hints everywhere"""
        (root / 'AGENTS.md').write_text(agents_content)

        # Note: We intentionally do NOT create .openhands/microagents directory
        yield root


def test_load_agents_md_without_microagents_dir(temp_dir_with_agents_md_only):
    """Test loading AGENTS.md file when .openhands/microagents directory doesn't exist."""
    # Try to load from non-existent microagents directory
    microagents_dir = temp_dir_with_agents_md_only / '.openhands' / 'microagents'

    repo_agents, knowledge_agents = load_microagents_from_dir(microagents_dir)

    # This should find the AGENTS.md file even though microagents_dir doesn't exist
    assert len(repo_agents) == 1  # Only AGENTS.md
    assert 'agents' in repo_agents
    assert len(knowledge_agents) == 0

    # Check AGENTS.md agent
    agents_agent = repo_agents['agents']
    assert isinstance(agents_agent, RepoMicroagent)
    assert agents_agent.name == 'agents'
    assert 'Install deps: `poetry install`' in agents_agent.content
    assert agents_agent.type == MicroagentType.REPO_KNOWLEDGE


@pytest.fixture
def temp_dir_with_both_cursorrules_and_agents():
    """Create a temporary directory with both .cursorrules and AGENTS.md files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create .cursorrules file
        cursorrules_content = """Always use Python for new files.
Follow PEP 8 style guidelines."""
        (root / '.cursorrules').write_text(cursorrules_content)

        # Create AGENTS.md file
        agents_content = """# Development Guide

## Setup commands

- Install deps: `poetry install`
- Run tests: `poetry run pytest`"""
        (root / 'AGENTS.md').write_text(agents_content)

        yield root


def test_load_both_cursorrules_and_agents_md(temp_dir_with_both_cursorrules_and_agents):
    """Test loading both .cursorrules and AGENTS.md files when .openhands/microagents doesn't exist."""
    # Try to load from non-existent microagents directory
    microagents_dir = (
        temp_dir_with_both_cursorrules_and_agents / '.openhands' / 'microagents'
    )

    repo_agents, knowledge_agents = load_microagents_from_dir(microagents_dir)

    # This should find both files
    assert len(repo_agents) == 2  # .cursorrules + AGENTS.md
    assert 'cursorrules' in repo_agents
    assert 'agents' in repo_agents
    assert len(knowledge_agents) == 0

    # Check both agents
    cursorrules_agent = repo_agents['cursorrules']
    assert isinstance(cursorrules_agent, RepoMicroagent)
    assert 'Always use Python for new files' in cursorrules_agent.content

    agents_agent = repo_agents['agents']
    assert isinstance(agents_agent, RepoMicroagent)
    assert 'Install deps: `poetry install`' in agents_agent.content


def test_dependency_repos_parsing():
    """Test parsing of dependency_repos from frontmatter."""
    from openhands.microagent import DependencyRepo, collect_dependency_repos

    # Create a microagent with dependency_repos
    agent_content = """---
name: test-agent
version: 1.0.0
dependency_repos:
  - name: frontend
    repo: OpenHands/OpenHands
  - name: backend
    repo: OpenHands/software-agent-sdk
---

# Test Agent

This agent has dependency repos.
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / 'test.md').write_text(agent_content)

        agent = BaseMicroagent.load(root / 'test.md', root)

        # Check that dependency_repos are parsed correctly
        assert len(agent.dependency_repos) == 2
        assert agent.dependency_repos[0].name == 'frontend'
        assert agent.dependency_repos[0].repo == 'OpenHands/OpenHands'
        assert agent.dependency_repos[1].name == 'backend'
        assert agent.dependency_repos[1].repo == 'OpenHands/software-agent-sdk'


def test_collect_dependency_repos():
    """Test collecting dependency repos from multiple microagents."""
    from openhands.microagent import DependencyRepo, collect_dependency_repos

    # Create microagents with dependency_repos
    agent1_content = """---
name: agent1
version: 1.0.0
dependency_repos:
  - name: frontend
    repo: OpenHands/OpenHands
---

# Agent 1
"""
    agent2_content = """---
name: agent2
version: 1.0.0
dependency_repos:
  - name: backend
    repo: OpenHands/software-agent-sdk
  - name: frontend
    repo: OpenHands/OpenHands
---

# Agent 2
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / 'agent1.md').write_text(agent1_content)
        (root / 'agent2.md').write_text(agent2_content)

        agent1 = BaseMicroagent.load(root / 'agent1.md', root)
        agent2 = BaseMicroagent.load(root / 'agent2.md', root)

        # Collect dependency repos (should deduplicate)
        dependency_repos = collect_dependency_repos([agent1, agent2])

        # Should have 2 unique repos (frontend appears twice but should be deduplicated)
        assert len(dependency_repos) == 2
        repo_names = {dep.repo for dep in dependency_repos}
        assert 'OpenHands/OpenHands' in repo_names
        assert 'OpenHands/software-agent-sdk' in repo_names


def test_microagent_without_dependency_repos():
    """Test that microagents without dependency_repos work correctly."""
    from openhands.microagent import collect_dependency_repos

    agent_content = """---
name: simple-agent
version: 1.0.0
---

# Simple Agent

No dependency repos here.
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / 'simple.md').write_text(agent_content)

        agent = BaseMicroagent.load(root / 'simple.md', root)

        # Check that dependency_repos is empty
        assert len(agent.dependency_repos) == 0

        # Collect should return empty list
        dependency_repos = collect_dependency_repos([agent])
        assert len(dependency_repos) == 0
