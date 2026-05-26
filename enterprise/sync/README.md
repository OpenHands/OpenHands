# Resend Sync Service

This service syncs users from the OpenHands database to a Resend.com audience. It runs as a Kubernetes CronJob that periodically queries the OpenHands database and adds any new users to the specified Resend audience.

## Features

- Syncs OpenHands users with email addresses to a Resend.com audience
- Handles rate limiting and retries with exponential backoff
- Runs as a Kubernetes CronJob
- Configurable batch size and sync frequency

## Configuration

The service is configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RESEND_API_KEY` | Resend API key | (required) |
| `RESEND_AUDIENCE_ID` | Resend audience ID | (required) |
| `BATCH_SIZE` | Number of users to process in each batch | `2000` |
| `MAX_RETRIES` | Maximum number of retries for API calls | `3` |
| `INITIAL_BACKOFF_SECONDS` | Initial backoff time for retries | `1` |
| `MAX_BACKOFF_SECONDS` | Maximum backoff time for retries | `60` |
| `BACKOFF_FACTOR` | Backoff factor for retries | `2` |
| `RATE_LIMIT` | Rate limit for API calls (requests per second) | `2` |


### Migration from Keycloak-backed sync

This sync no longer reads or counts users through Keycloak. Deployments should remove the old `KEYCLOAK_SERVER_URL`, `KEYCLOAK_REALM_NAME`, and `KEYCLOAK_ADMIN_PASSWORD` wiring for this CronJob and ensure it has the same OpenHands database configuration as the application server. If the database session cannot be created, the job fails before attempting to sync contacts.

Display-name personalization now comes from the local `user.git_user_name` field when present; users without a stored display name continue to receive the generic `Hi there,` greeting.

**Failure modes:** If the OpenHands database is unavailable, the sync job exits with status code 1 and logs the failure. No contacts are synced until the database connection is restored. Monitor the CronJob logs for database connection errors.

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
