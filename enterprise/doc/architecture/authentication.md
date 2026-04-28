# Authentication Flow (SaaS Deployment)

OpenHands uses Keycloak for identity management in the SaaS deployment. The authentication flow involves multiple services:

```mermaid
sequenceDiagram
    autonumber
    participant User as User (Browser)
    participant App as App Server
    participant KC as Keycloak
    participant IdP as Identity Provider<br/>(GitHub, Google, etc.)
    participant DB as User Database

    Note over User,DB: OAuth 2.0 / OIDC Authentication Flow

    User->>App: Access OpenHands
    App->>User: Redirect to Keycloak
    User->>KC: Login request
    KC->>User: Show login options
    User->>KC: Select provider (e.g., GitHub)
    KC->>IdP: OAuth redirect
    User->>IdP: Authenticate
    IdP-->>KC: OAuth callback + tokens
    Note over KC: Create/update user session
    KC-->>User: Redirect with auth code
    User->>App: Auth code
    App->>KC: Exchange code for tokens
    KC-->>App: Access token + Refresh token
    Note over App: Create signed JWT cookie
    App->>DB: Store/update user record
    App-->>User: Set keycloak_auth cookie

    Note over User,DB: Subsequent Requests

    User->>App: Request with cookie
    Note over App: Verify JWT signature
    App->>KC: Validate token (if needed)
    KC-->>App: Token valid
    Note over App: Extract user context
    App-->>User: Authorized response
```

### Authentication Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Keycloak** | Identity provider, SSO, token management | External service |
| **UserAuth** | Abstract auth interface | `openhands/server/user_auth/user_auth.py` |
| **SaasUserAuth** | Keycloak implementation | `enterprise/server/auth/saas_user_auth.py` |
| **JWT Service** | Token signing/verification | `openhands/app_server/services/jwt_service.py` |
| **Auth Routes** | Login/logout endpoints | `enterprise/server/routes/auth.py` |

### Token Flow

1. **Keycloak Access Token**: Short-lived token for API access
2. **Keycloak Refresh Token**: Long-lived token to obtain new access tokens
3. **Signed JWT Cookie**: App Server's session cookie containing encrypted Keycloak tokens
4. **Provider Tokens**: OAuth tokens for GitHub, GitLab, etc. (stored separately for git operations)
