# OpenHands Runtime

## Introduction

The OpenHands Runtime folder contains the core components responsible for executing actions and managing the runtime environment for the OpenHands project. This README provides an overview of the main components and their interactions.
You can learn more about how the runtime works in the [Docker Runtime](https://docs.all-hands.dev/usage/architecture/runtime) documentation.

## Main Components

### 1. base.py

The `base.py` file defines the `Runtime` class, which serves as the primary [interface](./base.py) for agent interactions with the external environment. It handles various operations including:

- Bash sandbox execution
- Browser interactions
- Filesystem operations
- Environment variable management
- Plugin management

Key features of the `Runtime` class:
- Initialization with configuration and event stream
- Asynchronous initialization (`ainit`) for setting up environment variables
- Action execution methods for different types of actions (run, read, write, browse, etc.)
- Abstract methods for file operations (to be implemented by subclasses)

### 2. impl/action_execution/action_execution_client.py
The `action_execution_client.py` file contains the `ActionExecutionClient` class, which implements the Runtime interface. It is an abstract implementation, meaning
it still needs to be extended by a concrete implementation to be used.

This client interacts with an action_execution_server (defined below) via HTTP
calls to actually perform runtime actions.

### 3. action_execution_server.py

The `action_executor_server.py` file contains the `ActionExecutor` class, which is responsible for executing actions received via the `/execute_action` HTTP endpoint. It returns observations in the HTTP response.

Key features of the `ActionExecutor` class:
- Initialization of user environment and bash shell
- Plugin management and initialization
- Execution of various action types (bash commands, IPython cells, file operations, browsing)
- Integration with BrowserEnv for web interactions

### 4. Other Implementations
The `./impl/` directory contains a few different Runtime implementations, all of
which extend the `ActionExecutionClient` class. These implementations
handle the lifecycle of a Docker container or other environment running the
ActionExecutor server.

There are currently four implementations:
* Docker (runs locally in a Docker container)
* Remote (runs via a custom HTTP API for creating, pausing, resuming, and stopping runtimes in a remote environment)
* Modal (uses the Modal API)
* Runloop (uses the Runloop API)

You may also add your own `Runtime` subclass to the classpath and configure it like this:

```toml
runtime = "app.my.CustomRuntime"
```

## Workflow Description

1. **Initialization**:
   - The `Runtime` is initialized with configuration and event stream.
   - Environment variables are set up using `ainit` method.
   - Plugins are loaded and initialized.

2. **Action Handling**:
   - The `Runtime` receives actions through the event stream.
   - Actions are validated and routed to appropriate execution methods.

3. **Action Execution**:
   - Different types of actions are executed:
     - Bash commands using `run` method
     - IPython cells using `run_ipython` method
     - File operations (read/write) using `read` and `write` methods
     - Web browsing using `browse` and `browse_interactive` methods

4. **Observation Generation**:
   - After action execution, corresponding observations are generated.
   - Observations are added to the event stream.

5. **Plugin Integration**:
   - Plugins like Jupyter and AgentSkills are initialized and integrated into the runtime.

6. **Sandbox Environment**:
   - The `ActionExecutor` sets up a sandboxed environment inside a Docker container.
   - User environment and bash shell are initialized.
   - Actions received from the OpenHands backend are executed in this sandboxed environment.

7. **Browser Interactions**:
   - Web browsing actions are handled using the `BrowserEnv` class.

## Important Notes

- The runtime uses asynchronous programming (asyncio) for efficient execution.
- Environment variables can be added dynamically to both IPython and Bash environments.
- File operations and command executions are abstracted, allowing for different implementations in subclasses.
- The system uses a plugin architecture for extensibility.
- All interactions with the external environment are managed through the Runtime, ensuring a controlled and secure execution environment.

## Runtime Types

### Docker Runtime

The Docker Runtime is designed for local execution using Docker containers:

- Creates and manages a Docker container for each session
- Executes actions within the container
- Supports direct file system access and local resource management
- Ideal for development, testing, and scenarios requiring full control over the execution environment

Key features:
- Real-time logging and debugging capabilities
- Direct access to the local file system
- Faster execution due to local resources
- Container isolation for security

This is the default runtime used within OpenHands.

### Local Runtime

The Local Runtime is designed for direct execution on the local machine. Currently only supports running as the local user:

- Runs the action_execution_server directly on the host
- No Docker container overhead
- Direct access to local system resources
- Ideal for development and testing when Docker is not available or desired

Key features:
- Minimal setup required
- Direct access to local resources
- No container overhead
- Fastest execution speed

**Important: This runtime provides no isolation as it runs directly on the host machine. All actions are executed with the same permissions as the user running OpenHands. For secure execution with proper isolation, use the Docker Runtime instead.**

### Remote Runtime

The Remote Runtime is designed for execution in a remote environment:

- Connects to a remote server running the ActionExecutor
- Executes actions by sending requests to the remote client
- Supports distributed execution and cloud-based deployments
- Ideal for production environments, scalability, and scenarios where local resource constraints are a concern

Key features:
- Scalability and resource flexibility
- Reduced local resource usage
- Support for cloud-based deployments
- Potential for improved security through isolation

At the time of this writing, this is mostly used in parallel evaluation, such as this example for [SWE-Bench](https://github.com/OpenHands/OpenHands/tree/main/evaluation/benchmarks/swe_bench#run-inference-on-remoteruntime-experimental).

## Related Components

- The runtime interacts closely with the event system defined in the `openhands.events` module.
- It relies on configuration classes from `openhands.core.config`.
- Logging is handled through `openhands.core.logger`.

This section provides an overview of the OpenHands Runtime folder. For more detailed information on specific components or usage, please refer to the individual files and their docstrings.
