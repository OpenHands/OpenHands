# Project-only skills loading tests
Feature: Skills loading respects project-only configuration
  In order to prevent unwanted context injection
  As the system
  I want to load only project skills when starting conversations

  Scenario: Conversation loads project skills only
    Given the agent-server is running with mock skills endpoint
    And project skills are available: "create_file", "run_tests"
    And global skills exist: "security_audit", "github_integration"
    And user skills exist: "personal_productivity"
    When a conversation starts
    Then the agent-server receives a skills loading request
    And the request has load_public=false
    And the request has load_user=false
    And the request has load_project=true
    And the request has load_org=false
    And the agent loads only: "create_file", "run_tests"
    And the agent does not load: "security_audit", "github_integration", "personal_productivity"

  Scenario: Conversation with selected repository loads project skills only
    Given the agent-server is running with mock skills endpoint
    And project "my-org/my-repo" has skills available: "repo_skill_1", "repo_skill_2"
    And global skills exist: "global_skill_1"
    When a conversation starts for repository "my-org/my-repo"
    Then the agent-server receives a skills loading request
    And the request specifies load_project=true
    And the request specifies load_public=false
    And the request specifies load_user=false
    And the request specifies load_org=false
    And conversation initialization completes successfully
    And the agent contains only project-specific skills

  Scenario: Multiple conversations load their respective project skills
    Given the agent-server is running with mock skills endpoint
    And project A has skills available: "skill_a1", "skill_a2"
    And project B has skills available: "skill_b1", "skill_b2"
    When conversation 1 starts for project A
    And conversation 2 starts for project B
    Then conversation 1 agent loads: "skill_a1", "skill_a2"
    And conversation 2 agent loads: "skill_b1", "skill_b2"
    And no skills bleed between conversations

  Scenario: Empty project skills results in empty agent skills
    Given the agent-server is running with mock skills endpoint
    And project has no skills defined
    When a conversation starts
    Then the agent-server receives a skills loading request
    And the request has load_project=true
    And the agent skills list is empty
    And agent remains functional with zero skills
