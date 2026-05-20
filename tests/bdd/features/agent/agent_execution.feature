Feature: Agent executes steps in response to user input
  In order to process user tasks
  As an agent
  I want to execute steps, call LLM, and invoke tools

  Scenario: User sends a message and agent responds
    Given the LLM is configured with default responses
    When the user sends a message
    Then the agent receives the message
    And the LLM is called with the user message
    And the agent returns a response with an action
