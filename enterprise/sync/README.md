# Resend Sync Service

This service syncs users from Keycloak to a Resend.com audience. It runs as a Kubernetes CronJob that periodically queries the Keycloak database and adds any new users to the specified Resend audience.

## Features

- Syncs users from Keycloak to Resend.com audience
- Handles rate limiting and retries with exponential backoff
- Runs as a Kubernetes CronJob
- Configurable batch size and sync frequency

## Configuration

The service is configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RESEND_API_KEY` | Resend API key | (required) |
| `RESEND_AUDIENCE_ID` | Resend audience ID | (required) |
| `KEYCLOAK_SERVER_URL` | URL of the Keycloak server | (required) |
| `KEYCLOAK_REALM_NAME` | Keycloak realm name | (required) |
| `KEYCLOAK_ADMIN_PASSWORD` | Keycloak admin password | (required) |
| `BATCH_SIZE` | Number of users to process in each batch | `100` |
| `MAX_RETRIES` | Maximum number of retries for API calls | `3` |
| `INITIAL_BACKOFF_SECONDS` | Initial backoff time for retries | `1` |
| `MAX_BACKOFF_SECONDS` | Maximum backoff time for retries | `60` |
| `BACKOFF_FACTOR` | Backoff factor for retries | `2` |
| `RATE_LIMIT` | Rate limit for API calls (requests per second) | `2` |

## Deployment

The service is deployed as part of the openhands Helm chart. To enable it, set the following in your values.yaml:

```yaml
resendSync:
  enabled: true
  audienceId: "your-audience-id"
```

### Prerequisites

- Kubernetes cluster with the openhands chart deployed
- Resend.com API key stored in a Kubernetes secret named `resend-api-key`
- Resend.com audience ID

## Running Manually

You can run the sync job manually by executing:

```bash
python -m app.sync.resend
```

Make sure all required environment variables are set before running the script.
