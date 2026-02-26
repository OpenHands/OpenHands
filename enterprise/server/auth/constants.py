import os

GITHUB_APP_CLIENT_ID = os.getenv('GITHUB_APP_CLIENT_ID', '').strip()
GITHUB_APP_CLIENT_SECRET = os.getenv('GITHUB_APP_CLIENT_SECRET', '').strip()
GITHUB_APP_WEBHOOK_SECRET = os.getenv('GITHUB_APP_WEBHOOK_SECRET', '')
GITHUB_APP_PRIVATE_KEY = os.getenv('GITHUB_APP_PRIVATE_KEY', '').replace('\\n', '\n')
KEYCLOAK_SERVER_URL = os.getenv('KEYCLOAK_SERVER_URL', '').rstrip('/')
KEYCLOAK_REALM_NAME = os.getenv('KEYCLOAK_REALM_NAME', '')
KEYCLOAK_PROVIDER_NAME = os.getenv('KEYCLOAK_PROVIDER_NAME', '')
KEYCLOAK_CLIENT_ID = os.getenv('KEYCLOAK_CLIENT_ID', '')
KEYCLOAK_CLIENT_SECRET = os.getenv('KEYCLOAK_CLIENT_SECRET', '')
KEYCLOAK_SERVER_URL_EXT = os.getenv(
    'KEYCLOAK_SERVER_URL_EXT', f'https://{os.getenv("AUTH_WEB_HOST", "")}'
).rstrip('/')
KEYCLOAK_ADMIN_PASSWORD = os.getenv('KEYCLOAK_ADMIN_PASSWORD', '')
GITLAB_APP_CLIENT_ID = os.getenv('GITLAB_APP_CLIENT_ID', '').strip()
GITLAB_APP_CLIENT_SECRET = os.getenv('GITLAB_APP_CLIENT_SECRET', '').strip()
BITBUCKET_APP_CLIENT_ID = os.getenv('BITBUCKET_APP_CLIENT_ID', '').strip()
BITBUCKET_APP_CLIENT_SECRET = os.getenv('BITBUCKET_APP_CLIENT_SECRET', '').strip()
ENABLE_ENTERPRISE_SSO = os.getenv('ENABLE_ENTERPRISE_SSO', '').strip()
ENABLE_JIRA = os.environ.get('ENABLE_JIRA', 'false') == 'true'
ENABLE_JIRA_DC = os.environ.get('ENABLE_JIRA_DC', 'false') == 'true'
ENABLE_LINEAR = os.environ.get('ENABLE_LINEAR', 'false') == 'true'
JIRA_CLIENT_ID = os.getenv('JIRA_CLIENT_ID', '').strip()
JIRA_CLIENT_SECRET = os.getenv('JIRA_CLIENT_SECRET', '').strip()
LINEAR_CLIENT_ID = os.getenv('LINEAR_CLIENT_ID', '').strip()
LINEAR_CLIENT_SECRET = os.getenv('LINEAR_CLIENT_SECRET', '').strip()
JIRA_DC_CLIENT_ID = os.getenv('JIRA_DC_CLIENT_ID', '').strip()
JIRA_DC_CLIENT_SECRET = os.getenv('JIRA_DC_CLIENT_SECRET', '').strip()
JIRA_DC_BASE_URL = os.getenv('JIRA_DC_BASE_URL', '').strip()
JIRA_DC_ENABLE_OAUTH = os.getenv('JIRA_DC_ENABLE_OAUTH', '1') in ('1', 'true')
AUTH_URL = os.getenv('AUTH_URL', '').rstrip('/')
ROLE_CHECK_ENABLED = os.getenv('ROLE_CHECK_ENABLED', 'false').lower() in (
    '1',
    'true',
    't',
    'yes',
    'y',
    'on',
)

DUPLICATE_EMAIL_CHECK = os.getenv('DUPLICATE_EMAIL_CHECK', 'true') in ('1', 'true')

# reCAPTCHA Enterprise
RECAPTCHA_PROJECT_ID = os.getenv('RECAPTCHA_PROJECT_ID', '').strip()
RECAPTCHA_SITE_KEY = os.getenv('RECAPTCHA_SITE_KEY', '').strip()
RECAPTCHA_HMAC_SECRET = os.getenv('RECAPTCHA_HMAC_SECRET', '').strip()
RECAPTCHA_BLOCK_THRESHOLD = float(os.getenv('RECAPTCHA_BLOCK_THRESHOLD', '0.3'))

# Account Defender labels that indicate suspicious activity
SUSPICIOUS_LABELS = {
    'SUSPICIOUS_LOGIN_ACTIVITY',
    'SUSPICIOUS_ACCOUNT_CREATION',
    'RELATED_ACCOUNTS_NUMBER_HIGH',
}
