# Security

Given the impressive capabilities of OpenHands and similar coding agents, ensuring robust security measures is essential to prevent unintended actions or security breaches. The SecurityAnalyzer framework provides a structured approach to monitor and analyze agent actions for potential security risks.

To enable this feature:
* From the web interface
    * Open Configuration (by clicking the gear icon in the bottom right)
    * Select a Security Analyzer from the dropdown
    * Save settings
    * (to disable) repeat the same steps, but click the X in the Security Analyzer dropdown
* From config.toml
```toml
[security]
# Enable confirmation mode
confirmation_mode = true
# The security analyzer to use
security_analyzer = "your-security-analyzer"
```
(to disable) remove the lines from config.toml

## SecurityAnalyzer Base Class

The `SecurityAnalyzer` class (analyzer.py) is an abstract base class designed to listen to an event stream and analyze actions for security risks and eventually act before the action is executed. Below is a detailed explanation of its components and methods:

### Initialization

- **event_stream**: An instance of `EventStream` that the analyzer will listen to for events.

### Event Handling

- **on_event(event: Event)**: Handles incoming events. If the event is an `Action`, it evaluates its security risk and acts upon it.

### Abstract Methods

- **handle_api_request(request: Request)**: Abstract method to handle API requests.
- **log_event(event: Event)**: Logs events.
- **act(event: Event)**: Defines actions to take based on the analyzed event.
- **security_risk(event: Action)**: Evaluates the security risk of an action and returns the risk level.
- **close()**: Cleanups resources used by the security analyzer.

In conclusion, a concrete security analyzer should evaluate the risk of each event and act accordingly (e.g. auto-confirm, send Slack message, etc).

For customization and decoupling from the OpenHands core logic, the security analyzer can define its own API endpoints that can then be accessed from the frontend. These API endpoints need to be secured (do not allow more capabilities than the core logic
provides).

## How to implement your own Security Analyzer

1. Create a submodule in [security](/openhands/security/) with your analyzer's desired name
    * Have your main class inherit from [SecurityAnalyzer](/openhands/security/analyzer.py)
    * Optional: define API endpoints for `/api/security/{path:path}` to manage settings,
2. Add your analyzer class to the [options](/openhands/security/options.py) to have it be visible from the frontend combobox
3. Optional: implement your modal frontend (for when you click on the lock) in [security](/frontend/src/components/modals/security/) and add your component to [Security.tsx](/frontend/src/components/modals/security/Security.tsx)

## Implemented Security Analyzers

### LLM Risk Analyzer (Default)

The LLM Risk Analyzer is the default security analyzer that leverages LLM-provided risk assessments. It respects the `security_risk` attribute that can be set by the LLM when generating actions, allowing for intelligent risk assessment based on the context and content of each action.

Features:

* Uses LLM-provided risk assessments (LOW, MEDIUM, HIGH)
* Automatically requires confirmation for HIGH-risk actions
* Respects confirmation mode settings for MEDIUM and LOW-risk actions
* Lightweight and efficient - no external dependencies
* Integrates seamlessly with the agent's decision-making process

The LLM Risk Analyzer checks if actions have a `security_risk` attribute set by the LLM and maps it to the appropriate `ActionSecurityRisk` level. If no risk assessment is provided, it defaults to UNKNOWN.

### Invariant

It uses the [Invariant Analyzer](https://github.com/invariantlabs-ai/invariant) to analyze traces and detect potential issues with OpenHands's workflow. It uses confirmation mode to ask for user confirmation on potentially risky actions.

This allows the agent to run autonomously without fear that it will inadvertently compromise security or perform unintended actions that could be harmful.

Features:

* Detects:
    * potential secret leaks by the agent
    * security issues in Python code
    * malicious bash commands
    * dangerous user tasks (browsing agent setting)
    * harmful content generation (browsing agent setting)
* Logs:
    * actions and their associated risk
    * OpenHands traces in JSON format
* Run-time settings:
    * the [invariant policy](https://github.com/invariantlabs-ai/invariant?tab=readme-ov-file#policy-language)
    * acceptable risk threshold
    * (Optional) check_browsing_alignment flag
    * (Optional) guardrail_llm that assesses if the agent behaviour is safe

Browsing Agent Safety:

* Guardrail feature that uses the underlying LLM of the agent to:
    * Examine the user's request and check if it is harmful.
    * Examine the content entered by the agent in a textbox (argument of the “fill” browser action) and check if it is harmful.

* If the guardrail evaluates either of the 2 conditions to be true, it emits a change_agent_state action and transforms the AgentState to ERROR. This stops the agent from proceeding further.

* To enable this feature: In the InvariantAnalyzer object, set the check_browsing_alignment attribute to True and initialize the guardrail_llm attribute with an LLM object.

### Gray Swan

The Gray Swan Security Analyzer integrates with [Gray Swan AI's Cygnal API](https://docs.grayswan.ai/cygnal/monitor-requests) to provide advanced AI safety monitoring for OpenHands agents.

#### Getting Started
To get started with the Gray Swan security analyzer (powered by Cygnal):

1. Existing Gray Swan customers should already have access to the platform.
2. New users should [request a demo](https://hubs.ly/Q03-sX2z0) to get onboarded and receive API credentials.
3. During onboarding, Gray Swan can also provide custom policy recommendations and integration support.
4. If you just want to use Cygnal's default protections, you can move to the next section.
5. If you want **even more** custom protection, you can create your own policy [here](https://platform.grayswan.ai/policies). Policies are composed of rules, which require a short title, e.g. "Git Operations", and then the rule itself, e.g. "The agent should never push code directly to the main branch".

#### OpenHands Configuration:

To use the GraySwan analyzer, set the following environment variables:

* `GRAYSWAN_API_KEY`: Your GraySwan API key (required)
* `GRAYSWAN_POLICY_ID`: Your GraySwan policy ID (optional)

Then configure OpenHands to use the GraySwan analyzer:

```toml
[security]
security_analyzer = "grayswan"
```

or select "grayswan" from the dropdown in settings!
