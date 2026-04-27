import os

GITLAB_HOST = os.environ.get('GITLAB_HOST', 'gitlab.com').strip() or 'gitlab.com'
