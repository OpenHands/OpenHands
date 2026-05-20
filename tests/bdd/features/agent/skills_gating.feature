# Environment variable gating tests
Feature: OPENHANDS_SKILLS_ENABLED environment variable controls skill loading
  In order to test with zero context noise
  As a developer
  I want to disable skills entirely via environment variable

  Scenario: Skills loading enabled by default
    Given OPENHANDS_SKILLS_ENABLED is not set
    And project has skills defined: "project_skill"
    When a conversation starts
    Then the agent-server is called for skills
    And skills are loaded from project
    And agent receives: "project_skill"

  Scenario: Skills loading enabled when OPENHANDS_SKILLS_ENABLED=true
    Given OPENHANDS_SKILLS_ENABLED='true'
    And project has skills defined: "my_skill"
    When a conversation starts
    Then the agent-server is called for skills
    And agent receives: "my_skill"

  Scenario: Skills loading enabled when OPENHANDS_SKILLS_ENABLED=1
    Given OPENHANDS_SKILLS_ENABLED='1'
    And project has skills defined: "my_skill"
    When a conversation starts
    Then the agent-server is called for skills
    And agent receives: "my_skill"

  Scenario: Skills loading disabled when OPENHANDS_SKILLS_ENABLED=false
    Given OPENHANDS_SKILLS_ENABLED='false'
    And project has skills defined: "project_skill"
    And agent-server would return skills
    When a conversation starts
    Then no HTTP request is made to agent-server /api/skills
    And load_and_merge_all_skills returns empty list
    And agent skills list is empty
    And agent remains fully functional
    And logs contain message: "Skills loading disabled by OPENHANDS_SKILLS_ENABLED env var"

  Scenario: Skills loading disabled when OPENHANDS_SKILLS_ENABLED=0
    Given OPENHANDS_SKILLS_ENABLED='0'
    And project has skills defined: "project_skill"
    When a conversation starts
    Then no HTTP request is made to agent-server
    And agent skills is empty
    And conversation proceeds normally

  Scenario: Env var is case-insensitive (FALSE)
    Given OPENHANDS_SKILLS_ENABLED='FALSE'
    And project has skills defined
    When a conversation starts
    Then agent skills is empty
    And skills loading disabled

  Scenario: Env var is case-insensitive (False)
    Given OPENHANDS_SKILLS_ENABLED='False'
    And project has skills defined
    When a conversation starts
    Then agent skills is empty

  Scenario: Whitespace is handled in env var value (leading)
    Given OPENHANDS_SKILLS_ENABLED=' true'
    And project has skills defined: "my_skill"
    When a conversation starts
    Then agent receives: "my_skill"

  Scenario: Whitespace is handled in env var value (trailing)
    Given OPENHANDS_SKILLS_ENABLED='true '
    And project has skills defined: "my_skill"
    When a conversation starts
    Then agent receives: "my_skill"

  Scenario: Whitespace is handled in env var value (both)
    Given OPENHANDS_SKILLS_ENABLED=' true '
    And project has skills defined: "my_skill"
    When a conversation starts
    Then agent receives: "my_skill"

  Scenario: Whitespace in false value
    Given OPENHANDS_SKILLS_ENABLED=' false '
    And project has skills defined
    When a conversation starts
    Then agent skills is empty
