"""GitLab configuration constants.

GITLAB_HOST is the hostname of the GitLab instance (e.g. 'gitlab.example.com').
Defaults to 'gitlab.com' for GitLab SaaS. Protocol prefixes and trailing
slashes are stripped, so 'https://gitlab.example.com/' is accepted.
"""

import os

GITLAB_HOST = (
    os.environ.get('GITLAB_HOST', 'gitlab.com')
    .strip()
    .removeprefix('https://')
    .removeprefix('http://')
    .rstrip('/')
    or 'gitlab.com'
)
