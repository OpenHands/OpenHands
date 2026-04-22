# Conversation Startup & WebSocket Flow

When a user starts a conversation, this sequence occurs:

```mermaid
sequenceDiagram
    autonumber
    participant User as User (Browser)
    participant App as App Server
    participant SS as Sandbox Service
    participant RAPI as Runtime API
    participant Pool as Warm Pool
    participant Sandbox as Sandbox (Container)
    participant AS as Agent Server
    participant AES as Action Execution Server

    Note over User,AES: Phase 1: Conversation Creation
    User->>App: POST /api/conversations
    Note over App: Authenticate user
    App->>SS: Create sandbox

    Note over SS,Pool: Phase 2: Runtime Provisioning
    SS->>RAPI: POST /start (image, env, config)
    RAPI->>Pool: Check for warm runtime
    alt Warm runtime available
        Pool-->>RAPI: Return warm runtime
        Note over RAPI: Assign to session
    else No warm runtime
        RAPI->>Sandbox: Create new container
        Sandbox->>AS: Start Agent Server
        Sandbox->>AES: Start Action Execution Server
        AES-->>AS: Ready
    end
    RAPI-->>SS: Runtime URL + session API key
    SS-->>App: Sandbox info
    App-->>User: Conversation ID + Sandbox URL

    Note over User,AES: Phase 3: Direct WebSocket Connection
    User->>AS: WebSocket: /sockets/events/{id}
    AS-->>User: Connection accepted
    AS->>User: Replay historical events

    Note over User,AES: Phase 4: User Sends Message
    User->>AS: WebSocket: SendMessageRequest
    Note over AS: Agent processes message
    Note over AS: LLM call → generate action

    Note over User,AES: Phase 5: Action Execution Loop
    loop Agent Loop
        AS->>AES: HTTP: Execute action
        Note over AES: Run in sandbox
        AES-->>AS: Observation result
        AS->>User: WebSocket: Event update
        Note over AS: Update state, next action
    end

    Note over User,AES: Phase 6: Task Complete
    AS->>User: WebSocket: AgentStateChanged (FINISHED)
```

### Key Points

1. **Initial Setup via App Server**: The App Server handles authentication and coordinates with the Sandbox Service
2. **Runtime API Provisioning**: The Sandbox Service calls the Runtime API, which checks for warm runtimes before creating new containers
3. **Warm Pool Optimization**: Pre-warmed runtimes reduce startup latency significantly
4. **Direct WebSocket to Sandbox**: Once created, the user's browser connects **directly** to the Agent Server inside the sandbox
5. **App Server Not in Hot Path**: After connection, all real-time communication bypasses the App Server entirely
6. **Agent Server Orchestrates**: The Agent Server manages the AI loop, calling the Action Execution Server for actual command execution
