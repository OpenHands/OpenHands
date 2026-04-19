# Agent Hub

In this folder, there may exist multiple implementations of `Agent` that will be used by the framework.

For example, `openhands/agenthub/codeact_agent`, etc.
Contributors from different backgrounds and interests can choose to contribute to any (or all!) of these directions.

## Constructing an Agent

The abstraction for an agent can be found [here](../controller/agent.py).

Agents are run inside of a loop. At each iteration, `agent.step()` is called with a
[State](../controller/state/state.py) input, and the agent must output an [Action](../events/action).

Every agent also has a `self.llm` which it can use to interact with the LLM configured by the user.
See the [LiteLLM docs for `self.llm.completion`](https://docs.litellm.ai/docs/completion).

## State

The `state` represents the running state of an agent in the OpenHands system. The class handles saving and restoring the agent session. It is serialized in a pickle.

The State object stores information about:

* Multi-agent state / delegates:
  * the 'root task' (conversation between the agent and the user)
  * the subtask (conversation between an agent and the user or another agent)
  * global and local iterations
  * delegate levels for multi-agent interactions
  * almost stuck state
* Running state of an agent:
  * current agent state (e.g., LOADING, RUNNING, PAUSED)
  * traffic control state for rate limiting
  * confirmation mode
  * the last error encountered
* History:
  * start and end IDs for events in agent's history. This allows to retrieve the actions taken by the agent, and observations (e.g. file content, command output) from the current or past sessions.
* Metrics:
  * global metrics for the current task
  * local metrics for the current subtask
* Extra data:
  * additional task-specific data

The agent can manage a structured task list through [`TaskTrackingAction`](../events/action/agent.py),
which powers the `task_tracker` tool used by agents such as `CodeActAgent`.

## Actions

Here is a list of available Actions, which can be returned by `agent.step()`:

- [`CmdRunAction`](../events/action/commands.py) - Runs a command inside a sandboxed terminal
- [`IPythonRunCellAction`](../events/action/commands.py) - Execute a block of Python code interactively (in Jupyter notebook) and receives `CmdOutputObservation`. Requires setting up `jupyter` [plugin](../runtime/plugins) as a requirement.
- [`FileReadAction`](../events/action/files.py) - Reads the content of a file
- [`FileWriteAction`](../events/action/files.py) - Writes new content to a file
- [`BrowseURLAction`](../events/action/browse.py) - Gets the content of a URL
- [`TaskTrackingAction`](../events/action/agent.py) - Views or updates the task list used for task management.
- [`AgentFinishAction`](../events/action/agent.py) - Stops the control loop, allowing the user/delegator agent to enter a new task
- [`AgentRejectAction`](../events/action/agent.py) - Stops the control loop, allowing the user/delegator agent to enter a new task
- [`AgentFinishAction`](../events/action/agent.py) - Stops the control loop, allowing the user to enter a new task
- [`MessageAction`](../events/action/message.py) - Represents a message from an agent or the user

To serialize and deserialize an action, you can use:
- `action.to_dict()` to serialize the action to a dictionary to be sent to the UI, including a user-friendly string representation of the message
- `action.to_memory()` to serialize the action to a dictionary to be sent to the LLM. It may include raw information, such as the underlying exceptions that occurred during the action execution.
- `action_from_dict(action_dict)` to deserialize the action from a dictionary.

## Observations

There are also several types of Observations. These are typically available in the step following the corresponding Action.
But they may also appear as a result of asynchronous events (e.g. a message from the user).

Here is a list of available Observations:

- [`CmdOutputObservation`](../events/observation/commands.py)
- [`BrowserOutputObservation`](../events/observation/browse.py)
- [`FileReadObservation`](../events/observation/files.py)
- [`FileWriteObservation`](../events/observation/files.py)
- [`TaskTrackingObservation`](../events/observation/task_tracking.py)
- [`ErrorObservation`](../events/observation/error.py)
- [`SuccessObservation`](../events/observation/success.py)

You can use `observation.to_dict()` and `observation_from_dict` to serialize and deserialize observations.

## Interface

Every agent must implement the following methods:

### `step`

```
def step(self, state: "State") -> "Action"
```

`step` moves the agent forward one step towards its goal. This probably means
sending a prompt to the LLM, then parsing the response into an `Action`.

## Agent Delegation

OpenHands is a multi-agentic system. Agents can delegate tasks to other agents, whether
prompted by the user, or when the agent decides to ask another agent for help. For example,
the `CodeActAgent` might delegate to the `BrowsingAgent` to answer questions that involve browsing
the web. The Delegator Agent forwards tasks to micro-agents, such as 'RepoStudyAgent' to study a repo,
or 'VerifierAgent' to verify a task completion.

### Understanding the terminology

A `task` is an end-to-end conversation between OpenHands (the whole system) and the user,
which might involve one or more inputs from the user. It starts with an initial input
(typically a task statement) from the user, and ends with either an `AgentFinishAction`
initiated by the agent, a stop initiated by the user, or an error.

A `subtask` is an end-to-end conversation between an agent and the user, or
another agent. If a `task` is conducted by a single agent, then it's also a `subtask`
itself. Otherwise, a `task` consists of multiple `subtasks`, each executed by
one agent.

For example, considering a task from the user: `tell me how many GitHub stars
OpenHands repo has`. Let's assume the default agent is CodeActAgent.

```
-- TASK STARTS (SUBTASK 0 STARTS) --

DELEGATE_LEVEL 0, ITERATION 0, LOCAL_ITERATION 0
CodeActAgent: I should request help from BrowsingAgent

-- DELEGATE STARTS (SUBTASK 1 STARTS) --

DELEGATE_LEVEL 1, ITERATION 1, LOCAL_ITERATION 0
BrowsingAgent: Let me find the answer on GitHub

DELEGATE_LEVEL 1, ITERATION 2, LOCAL_ITERATION 1
BrowsingAgent: I found the answer, let me convey the result and finish

-- DELEGATE ENDS (SUBTASK 1 ENDS) --

DELEGATE_LEVEL 0, ITERATION 3, LOCAL_ITERATION 1
CodeActAgent: I got the answer from BrowsingAgent, let me convey the result
and finish

-- TASK ENDS (SUBTASK 0 ENDS) --
```

Note how ITERATION counter is shared across agents, while LOCAL_ITERATION
is local to each subtask.
