# Agent Execution & LLM Flow

When the agent executes inside the sandbox, it makes LLM calls through LiteLLM:

```mermaid
sequenceDiagram
    autonumber
    participant User as User (Browser)
    participant AS as Agent Server
    participant Agent as Agent<br/>(CodeAct)
    participant LLM as LLM Class
    participant Lite as LiteLLM
    participant Proxy as LLM Proxy<br/>(llm-proxy.app.all-hands.dev)
    participant Provider as LLM Provider<br/>(OpenAI, Anthropic, etc.)
    participant AES as Action Execution Server

    Note over User,AES: Agent Loop - LLM Call Flow

    User->>AS: WebSocket: User message
    AS->>Agent: Process message
    Note over Agent: Build prompt from state

    Agent->>LLM: completion(messages, tools)
    Note over LLM: Apply config (model, temp, etc.)

    alt Using OpenHands Provider
        LLM->>Lite: litellm_proxy/{model}
        Lite->>Proxy: POST /chat/completions
        Note over Proxy: Auth, rate limit, routing
        Proxy->>Provider: Forward request
        Provider-->>Proxy: Response
        Proxy-->>Lite: Response
    else Using Direct Provider
        LLM->>Lite: {provider}/{model}
        Lite->>Provider: Direct API call
        Provider-->>Lite: Response
    end

    Lite-->>LLM: ModelResponse
    Note over LLM: Track metrics (cost, tokens)
    LLM-->>Agent: Parsed response

    Note over Agent: Parse action from response
    AS->>User: WebSocket: Action event

    Note over User,AES: Action Execution

    AS->>AES: HTTP: Execute action
    Note over AES: Run command/edit file
    AES-->>AS: Observation
    AS->>User: WebSocket: Observation event

    Note over Agent: Update state
    Note over Agent: Loop continues...
```

### LLM Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **LLM Class** | Wrapper with retries, metrics, config | `openhands/llm/llm.py` |
| **LiteLLM** | Universal LLM API adapter | External library |
| **LLM Proxy** | OpenHands managed proxy for billing/routing | `llm-proxy.app.all-hands.dev` |
| **LLM Registry** | Manages multiple LLM instances | `openhands/llm/llm_registry.py` |

### Model Routing

```
User selects model
        │
        ▼
┌───────────────────┐
│ Model prefix?     │
└───────────────────┘
        │
        ├── openhands/claude-3-5  ──► Rewrite to litellm_proxy/claude-3-5
        │                              Base URL: llm-proxy.app.all-hands.dev
        │
        ├── anthropic/claude-3-5  ──► Direct to Anthropic API
        │                              (User's API key)
        │
        ├── openai/gpt-4          ──► Direct to OpenAI API
        │                              (User's API key)
        │
        └── azure/gpt-4           ──► Direct to Azure OpenAI
                                       (User's API key + endpoint)
```

### LLM Proxy

When using `openhands/` prefixed models, requests are routed through a managed proxy.
See the [OpenHands documentation](https://docs.openhands.dev/) for details on supported models.
