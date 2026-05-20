@fast
Feature: OpenHands system prompt content is absent from nue agent context
  In order to maintain nue's custom identity and avoid injecting OpenHands guidance
  As the nue platform
  I want the agent system prompt to exclude all OpenHands-specific branding and instructions

  @fast
  Scenario: Default agent system prompt has no OpenHands identity claim
    Given the server builds the system prompt for a default non-planning agent
    Then the system prompt does not contain "You are OpenHands agent"

  @fast
  Scenario: Default agent system prompt has no AGENTS.md memory instruction
    Given the server builds the system prompt for a default non-planning agent
    Then the system prompt does not contain "AGENTS.md"

  @fast
  Scenario: Default agent system prompt has no OpenHands commit identity
    Given the server builds the system prompt for a default non-planning agent
    Then the system prompt does not contain "openhands@all-hands.dev"

  @fast
  Scenario: Default agent system prompt has no OpenHands documentation link
    Given the server builds the system prompt for a default non-planning agent
    Then the system prompt does not contain "docs.openhands.dev"
