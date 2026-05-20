# Error handling and resilience tests
Feature: Skills loading gracefully handles errors
  In order to maintain system stability
  As the system
  I want to handle missing/misconfigured skills gracefully

  Scenario: Agent-server returns 500 error on skills request
    Given the agent-server is configured to return 500 error
    And project has skills defined
    When a conversation starts
    Then load_and_merge_all_skills returns empty list
    And no exception is raised to caller
    And conversation initialization continues normally
    And logs contain warning about agent-server error
    And agent proceeds with zero skills

  Scenario: Agent-server returns 404 on skills endpoint
    Given the agent-server /api/skills endpoint returns 404
    And project has skills defined
    When a conversation starts
    Then load_and_merge_all_skills returns empty list
    And conversation initialization completes
    And agent is functional with zero skills
    And warning is logged

  Scenario: Agent-server request times out
    Given the agent-server skills request times out (60 second timeout)
    And project has skills defined
    When a conversation starts
    Then load_and_merge_all_skills returns empty list
    And timeout is handled gracefully
    And conversation proceeds normally
    And logs contain timeout warning

  Scenario: Agent-server returns invalid JSON
    Given the agent-server returns invalid JSON response
    And project has skills defined
    When a conversation starts
    Then load_and_merge_all_skills returns empty list
    And conversation initialization succeeds
    And warning is logged about malformed response

  Scenario: Agent-server returns response with missing required fields
    Given the agent-server returns skills missing required fields
    And valid skills are in the response
    When a conversation starts
    Then invalid skills are skipped
    And valid skills are loaded
    And warning logged for each invalid skill
    And agent loads only the valid skills

  Scenario: Agent-server returns empty skills list
    Given the agent-server returns empty skills list
    And project directory exists
    When a conversation starts
    Then load_and_merge_all_skills returns empty list
    And agent.skills is empty
    And agent remains functional
    And conversation proceeds normally

  Scenario: Project directory does not exist
    Given the agent-server skills endpoint is configured
    And project directory does not exist
    When a conversation starts
    Then request is sent to agent-server with project_dir path
    And agent-server handles missing directory gracefully
    And conversation proceeds
    And skills loading request completes

  Scenario: Agent-server returns skills with unexpected data types
    Given the agent-server returns skill with non-string name
    And other fields have unexpected types
    When a conversation starts
    Then invalid skills are skipped
    And valid skills are loaded
    And conversation continues normally
    And warnings logged for type errors

  Scenario: Very large skills response is handled
    Given the agent-server returns 1000 skills
    When a conversation starts
    Then all skills are processed
    And agent is initialized with all skills
    And no timeout or memory error occurs
    And conversation proceeds normally

  Scenario: Concurrent skill loading requests
    Given multiple conversations starting simultaneously
    And each requesting project skills
    When all conversations start concurrently
    Then each receives their own skills
    And no race conditions occur
    And all conversations initialize successfully
