# OpenHands Server

This is a WebSocket server that executes tasks using an agent.

## Recommended Prerequisites

- [Initialize the frontend code](../../frontend/README.md)
- Install Python 3.12 (`brew install python` for those using homebrew)
- Install pipx: (`brew install pipx` followed by `pipx ensurepath`)
- Install poetry: (`pipx install poetry`)

## Install

First build a distribution of the frontend code (From the project root directory):

```sh
cd frontend
npm install
npm run build
cd ..
```

Next run `poetry shell` (So you don't have to repeat `poetry run`)

## Start the Server

```sh
uvicorn openhands.server.listen:app --reload --port 3000
```

## Test the Server

You can use [`websocat`](https://github.com/vi/websocat) to test the server.

```sh
websocat ws://127.0.0.1:3000/ws
{"action": "start", "args": {"task": "write a bash script that prints hello"}}
```

## Supported Environment Variables

```sh
LLM_API_KEY=sk-... # Your Anthropic API Key
LLM_MODEL=claude-3-5-sonnet-20241022 # Default model for the agent to use
SANDBOX_VOLUMES=/path/to/your/workspace:/workspace:rw # Mount paths in format host_path:container_path:mode
SSL_VERIFY=false # Disable SSL certificate verification for self-hosted services (default: true)
INSECURE_SKIP_VERIFY=true # Alternative way to disable SSL verification (default: false)
```

## API Schema

There are two types of messages that can be sent to, or received from, the server:

* Actions
* Observations

### Actions

An action has three parts:

* `action`: The action to be taken
* `args`: The arguments for the action
* `message`: A friendly message that can be put in the chat log

There are several kinds of actions. Their arguments are listed below.
This list may grow over time.

* `initialize` - initializes the agent. Only sent by client.
  * `model` - the name of the model to use
  * `directory` - the path to the workspace
  * `agent_cls` - the class of the agent to use
* `start` - starts a new development task. Only sent by the client.
  * `task` - the task to start
* `read` - reads the content of a file.
  * `path` - the path of the file to read
* `write` - writes the content to a file.
  * `path` - the path of the file to write
  * `content` - the content to write to the file
* `run` - runs a command.
  * `command` - the command to run
* `browse` - opens a web page.
  * `url` - the URL to open
* `think` - Allows the agent to make a plan, set a goal, or record thoughts
  * `thought` - the thought to record
* `finish` - agent signals that the task is completed

### Observations

An observation has four parts:

* `observation`: The observation type
* `content`: A string representing the observed data
* `extras`: additional structured data
* `message`: A friendly message that can be put in the chat log

There are several kinds of observations. Their extras are listed below.
This list may grow over time.

* `read` - the content of a file
  * `path` - the path of the file read
* `browse` - the HTML content of a url
  * `url` - the URL opened
* `run` - the output of a command
  * `command` - the command run
  * `exit_code` - the exit code of the command
* `chat` - a message from the user

## Server Components

The following section describes the server-side components of the OpenHands project.

### 1. session/session.py

The `session.py` file defines the `Session` class, which represents a WebSocket session with a client. Key features include:

- Handling WebSocket connections and disconnections
- Initializing and managing the agent session
- Dispatching events between the client and the agent
- Sending messages and errors to the client

### 2. session/agent_session.py

The `agent_session.py` file contains the `AgentSession` class, which manages the lifecycle of an agent within a session. Key features include:

- Creating and managing the runtime environment
- Initializing the agent controller
- Handling security analysis
- Managing the event stream

### 3. session/conversation_manager/conversation_manager.py

The `conversation_manager.py` file defines the `ConversationManager` class, which is responsible for managing multiple client conversations. Key features include:

- Adding and restarting conversations
- Sending messages to specific conversations
- Cleaning up inactive conversations

### 4. listen.py

The `listen.py` file is the main server file that sets up the FastAPI application and defines various API endpoints. Key features include:

- Setting up CORS middleware
- Handling WebSocket connections
- Managing file uploads
- Providing API endpoints for agent interactions, file operations, and security analysis
- Serving static files for the frontend

## Workflow Description

1. **Server Initialization**:
   - The FastAPI application is created and configured in `listen.py`.
   - CORS middleware and static file serving are set up.
   - The `ConversationManager` is initialized.

2. **Client Connection**:
   - When a client connects via WebSocket, a new `Session` is created or an existing one is restarted.
   - The `Session` initializes an `AgentSession`, which sets up the runtime environment and agent controller.

3. **Agent Initialization**:
   - The client sends an initialization request.
   - The server creates and configures the agent based on the provided parameters.
   - The runtime environment is set up, and the agent controller is initialized.

4. **Event Handling**:
   - The `Session` manages the event stream between the client and the agent.
   - Events from the client are dispatched to the agent.
   - Observations from the agent are sent back to the client.

5. **File Operations**:
   - The server handles file uploads, ensuring they meet size and type restrictions.
   - File read and write operations are performed through the runtime environment.

6. **Security Analysis**:
   - If configured, a security analyzer is initialized for each session.
   - Security-related API requests are forwarded to the security analyzer.

7. **Session Management**:
   - The `ConversationManager` periodically cleans up inactive sessions.
   - It also handles sending messages to specific sessions when needed.

8. **API Endpoints**:
   - Various API endpoints are provided for agent interactions, file operations, and retrieving configuration defaults.

This server architecture allows for managing multiple client sessions, each with its own agent instance, runtime environment, and security analyzer. The event-driven design facilitates real-time communication between clients and agents, while the modular structure allows for easy extension and maintenance of different components.
