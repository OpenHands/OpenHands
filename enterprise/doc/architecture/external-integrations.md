# External Integrations

OpenHands integrates with external services (GitHub, Slack, Jira, etc.) through webhook-based event handling:

```mermaid
sequenceDiagram
    autonumber
    participant Ext as External Service<br/>(GitHub/Slack/Jira)
    participant App as App Server
    participant IntRouter as Integration Router
    participant Manager as Integration Manager
    participant Conv as Conversation Service
    participant Sandbox as Sandbox

    Note over Ext,Sandbox: Webhook Event Flow (e.g., GitHub Issue Created)

    Ext->>App: POST /api/integration/{service}/events
    App->>IntRouter: Route to service handler
    Note over IntRouter: Verify signature (HMAC)

    IntRouter->>Manager: Parse event payload
    Note over Manager: Extract context (repo, issue, user)
    Note over Manager: Map external user → OpenHands user

    Manager->>Conv: Create conversation (with issue context)
    Conv->>Sandbox: Provision sandbox
    Sandbox-->>Conv: Ready

    Manager->>Sandbox: Start agent with task

    Note over Ext,Sandbox: Agent Works on Task...

    Sandbox-->>Manager: Task complete
    Manager->>Ext: POST result<br/>(PR, comment, etc.)

    Note over Ext,Sandbox: Callback Flow (Agent → External Service)

    Sandbox->>App: Webhook callback<br/>/api/v1/webhooks
    App->>Manager: Process callback
    Manager->>Ext: Update external service
```

### Supported Integrations

| Integration | Trigger Events | Agent Actions |
|-------------|----------------|---------------|
| **GitHub** | Issue created, PR opened, @mention | Create PR, comment, push commits |
| **GitLab** | Issue created, MR opened | Create MR, comment, push commits |
| **Slack** | @mention in channel | Reply in thread, create tasks |
| **Jira** | Issue created/updated | Update ticket, add comments |
| **Linear** | Issue created | Update status, add comments |

### Integration Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Integration Routes** | Webhook endpoints per service | `enterprise/server/routes/integration/` |
| **Integration Managers** | Business logic per service | `enterprise/integrations/{service}/` |
| **Token Manager** | Store/retrieve OAuth tokens | `enterprise/server/auth/token_manager.py` |
| **Callback Processor** | Handle agent → service updates | `enterprise/integrations/{service}/*_callback_processor.py` |

### Integration Authentication

```
External Service (e.g., GitHub)
        │
        ▼
┌─────────────────────────────────┐
│ GitHub App Installation         │
│ - Webhook secret for signature  │
│ - App private key for API calls │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ User Account Linking            │
│ - Keycloak user ID              │
│ - GitHub user ID                │
│ - Stored OAuth tokens           │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Agent Execution                 │
│ - Uses linked tokens for API    │
│ - Can push, create PRs, comment │
└─────────────────────────────────┘
```
