# Context isolation tests - ensure no global/user/org skills leak
Feature: Skills context is properly isolated between projects
  In order to maintain clean agent context
  As a project
  I want to ensure my skills don't expose global/user/org context

  Scenario: Agent receives only project skills in context
    Given the agent-server is running with mock skills endpoint
    And project skills are: "project_skill_1", "project_skill_2"
    And global skills database contains: "global_skill_1", "global_skill_2"
    And user personal skills are: "user_skill_1"
    And org skills are: "org_skill_1"
    When an agent is initialized for the project
    Then agent skills contains exactly: "project_skill_1", "project_skill_2"
    And agent skills does not contain: "global_skill_1", "global_skill_2"
    And agent skills does not contain: "user_skill_1"
    And agent skills does not contain: "org_skill_1"
    And agent remains fully functional

  Scenario: No global skills loaded when load_public=false
    Given the agent-server is running with mock skills endpoint
    And project has no skills
    And global skills database contains: "global1", "global2"
    When a conversation starts
    Then the request has load_public=false
    And agent does not contain: "global1", "global2"

  Scenario: No user skills loaded when load_user=false
    Given the agent-server is running with mock skills endpoint
    And user personal skills are: "user1", "user2"
    And project has no skills
    When a conversation starts
    Then the request has load_user=false
    And agent does not contain: "user1", "user2"

  Scenario: No org skills loaded when load_org=false
    Given the agent-server is running with mock skills endpoint
    And org skills are: "org1", "org2"
    And project has no skills
    When a conversation starts
    Then the request has load_org=false
    And agent does not contain: "org1", "org2"

  Scenario: Project skills preserved when global/user/org would normally load
    Given the agent-server is running with mock skills endpoint
    And project skills are: "my_project_skill"
    And global skills would normally include: "global_skill"
    And user skills would normally include: "user_skill"
    And org skills would normally include: "org_skill"
    When a conversation starts
    Then the agent contains: "my_project_skill"
    And the agent does not contain: "global_skill", "user_skill", "org_skill"
    And context is completely isolated
