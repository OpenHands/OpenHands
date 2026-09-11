"""Event name constants for PostHog analytics.

Naming convention: PostHog recommended object-action, lowercase with spaces.
"""

# Phase 1 events
USER_LOGGED_IN = 'user logged in'

# Phase 2 events
USER_SIGNED_UP = 'user signed up'
CONVERSATION_CREATED = 'conversation created'
CONVERSATION_FINISHED = 'conversation finished'
CONVERSATION_ERRORED = 'conversation errored'
CONVERSATION_DELETED = 'conversation deleted'
CREDIT_PURCHASED = 'credit purchased'
CREDIT_LIMIT_REACHED = 'credit limit reached'

# Phase 4 events
GIT_PROVIDER_CONNECTED = 'git provider connected'
ONBOARDING_COMPLETED = 'onboarding completed'
SETTINGS_SAVED = 'settings saved'
TRAJECTORY_DOWNLOADED = 'trajectory downloaded'
TEAM_MEMBERS_INVITED = 'team members invited'

# Phase 5 events — integration & product-usage signals that the HubSpot
# contact sync previously derived from the database. Emitting them here lets
# PostHog aggregate the same metrics directly off the event stream, so the
# PostHog HubSpot destination can keep contacts in sync without the cron job.
API_KEY_CREATED = 'api key created'
CLI_DEVICE_LINKED = 'cli device linked'
PULL_REQUEST_CLOSED = 'pull request closed'
PULL_REQUEST_CREATED = 'pull request created'
PULL_REQUEST_MERGED = 'pull request merged'
SLACK_INTEGRATION_ENABLED = 'slack integration enabled'
JIRA_INTEGRATION_ENABLED = 'jira integration enabled'
