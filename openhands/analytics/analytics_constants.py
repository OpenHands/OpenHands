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
