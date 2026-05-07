# OpenHands Glossary

### Agent
The core AI entity in OpenHands that can perform software development tasks by interacting with tools, browsing the web, and modifying code.

#### Agent Controller
A component that manages the agent's lifecycle, handles its state, and coordinates interactions between the agent and various tools.

#### Agent Delegation
The ability of an agent to hand off specific tasks to other specialized agents for better task completion.

#### Agent Hub
A central registry of different agent types and their capabilities, allowing for easy agent selection and instantiation.

#### Agent Skill
A specific capability or function that an agent can perform, such as file manipulation, web browsing, or code editing.

#### Agent State
The current context and status of an agent, including its memory, active tools, and ongoing tasks.

#### CodeAct Agent
[A generalist agent in OpenHands](https://arxiv.org/abs/2407.16741) designed to perform tasks by editing and executing code.

### Browser
A system for web-based interactions and tasks.

#### Browser Gym
A testing and evaluation environment for browser-based agent interactions and tasks.

#### Web Browser Tool
A tool that enables agents to interact with web pages and perform web-based tasks.

### Commands
Terminal and execution related functionality.

#### Bash Session
A persistent terminal session that maintains state and history for bash command execution.
This uses tmux under the hood.

### Configuration
System-wide settings and options.

#### Agent Configuration
Settings that define an agent's behavior, capabilities, and limitations, including available tools and runtime settings.

#### Configuration Options
Settings that control various aspects of OpenHands behavior, including runtime, security, and agent settings.

#### LLM Config
Configuration settings for language models used by agents, including model selection and parameters.

#### LLM Draft Config
Settings for draft mode operations with language models, typically used for faster, lower-quality responses.

#### Runtime Configuration
Settings that define how the runtime environment should be set up and operated.

#### Security Options
Configuration settings that control security features and restrictions.

### Conversation
A sequence of interactions between a user and an agent, including messages, actions, and their results.

#### Conversation Info
Metadata about a conversation, including its status, participants, and timeline.

#### Conversation Manager
A component that handles the creation, storage, and retrieval of conversations.

#### Conversation Metadata
Additional information about conversations, such as tags, timestamps, and related resources.

#### Conversation Status
The current state of a conversation, including whether it's active, completed, or failed.

#### Conversation Store
A storage system for maintaining conversation history and related data.

### Events

#### Event
Every Conversation comprises a series of Events. Each Event is either an Action or an Observation.

#### Event Stream
A continuous flow of events that represents the ongoing activities and interactions in the system.

#### Action
A specific operation or command that an agent executes through available tools, such as running a command or editing a file.

#### Observation
The response or result returned by a tool after an agent's action, providing feedback about the action's outcome.

### Interface
Different ways to interact with OpenHands.

#### CLI Mode
A command-line interface mode for interacting with OpenHands agents without a graphical interface.

#### GUI Mode
A graphical user interface mode for interacting with OpenHands agents through a web interface.

#### Headless Mode
A mode of operation where OpenHands runs without a user interface, suitable for automation and scripting.

### Agent Memory
The system that decides which parts of the Event Stream (i.e. the conversation history) should be passed into each LLM prompt.

#### Memory Store
A storage system for maintaining agent memory and context across sessions.

#### Condenser
A component that processes and summarizes conversation history to maintain context while staying within token limits.

#### Truncation
A very simple Condenser strategy. Reduces conversation history or content to stay within token limits.

### Microagent
A specialized prompt that enhances OpenHands with domain-specific knowledge, repository-specific context, and task-specific workflows.

#### Microagent Registry
A central repository of available microagents and their configurations.

#### Public Microagent
A general-purpose microagent available to all OpenHands users, triggered by specific keywords. Located in `skills/`.

#### Repository Microagent
A type of microagent that provides repository-specific context and guidelines, stored in the `.openhands/microagents/` directory.

### Prompt
Components for managing and processing prompts.

#### Prompt Caching
A system for caching and reusing common prompts to improve performance.

#### Prompt Manager
A component that handles the loading, processing, and management of prompts used by agents, including microagents.

#### Response Parsing
The process of interpreting and structuring responses from language models and tools.

### Runtime
The execution environment where agents perform their tasks, which can be local, remote, or containerized.

#### Action Execution Server
A REST API that receives agent actions (e.g. bash commands, python code, browsing actions), executes them in the runtime environment, and returns the results.

#### Action Execution Client
A component that handles the execution of actions in the runtime environment, managing the communication between the agent and the runtime.

#### Docker Runtime
A containerized runtime environment that provides isolation and reproducibility for agent operations.

#### E2B Runtime
A specialized runtime environment built on E2B for secure and isolated code execution.

#### Local Runtime
A runtime environment that executes on the local machine, suitable for development and testing.

#### Modal Runtime
A runtime environment built on Modal for scalable and distributed agent operations.

#### Remote Runtime
A sandboxed environment that executes code and commands remotely, providing isolation and security for agent operations.

#### Runtime Builder
A component that builds a Docker image for the Action Execution Server based on a user-specified base image.

### Security
Security-related components and features.

#### Security Analyzer
A component that checks agent actions for potential security risks.
