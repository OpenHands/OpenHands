import os

GITLAB_HOST = (
    os.environ.get('GITLAB_HOST', 'gitlab.com')
    .strip()
    .removeprefix('https://')
    .removeprefix('http://')
    .rstrip('/')
    or 'gitlab.com'
)
