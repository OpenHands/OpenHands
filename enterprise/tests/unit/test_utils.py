from integrations.utils import (
    has_exact_mention,
    infer_repo_from_message,
    markdown_to_jira_markup,
)


def test_has_exact_mention():
    # Test basic exact match
    assert has_exact_mention('Hello @openhands!', '@openhands') is True
    assert has_exact_mention('@openhands at start', '@openhands') is True
    assert has_exact_mention('End with @openhands', '@openhands') is True
    assert has_exact_mention('@openhands', '@openhands') is True

    # Test no match
    assert has_exact_mention('No mention here', '@openhands') is False
    assert has_exact_mention('', '@openhands') is False

    # Test partial matches (should be False)
    assert has_exact_mention('Hello @openhands-agent!', '@openhands') is False
    assert has_exact_mention('Email: user@openhands.com', '@openhands') is False
    assert has_exact_mention('Text@openhands', '@openhands') is False
    assert has_exact_mention('@openhandsmore', '@openhands') is False

    # Test with special characters in mention
    assert has_exact_mention('Hi @open.hands!', '@open.hands') is True
    assert has_exact_mention('Using @open-hands', '@open-hands') is True
    assert has_exact_mention('With @open_hands_ai', '@open_hands_ai') is True

    # Test case insensitivity (function now handles case conversion internally)
    assert has_exact_mention('Hi @OpenHands', '@OpenHands') is True
    assert has_exact_mention('Hi @OpenHands', '@openhands') is True
    assert has_exact_mention('Hi @openhands', '@OpenHands') is True
    assert has_exact_mention('Hi @OPENHANDS', '@openhands') is True

    # Test multiple mentions
    assert has_exact_mention('@openhands and @openhands again', '@openhands') is True
    assert has_exact_mention('@openhands-agent and @openhands', '@openhands') is True

    # Test with surrounding punctuation
    assert has_exact_mention('Hey, @openhands!', '@openhands') is True
    assert has_exact_mention('(@openhands)', '@openhands') is True
    assert has_exact_mention('@openhands: hello', '@openhands') is True
    assert has_exact_mention('@openhands? yes', '@openhands') is True


def test_markdown_to_jira_markup():
    test_cases = [
        ('**Bold text**', '*Bold text*'),
        ('__Bold text__', '*Bold text*'),
        ('*Italic text*', '_Italic text_'),
        ('_Italic text_', '_Italic text_'),
        ('**Bold** and *italic*', '*Bold* and _italic_'),
        ('Mixed *italic* and **bold** text', 'Mixed _italic_ and *bold* text'),
        ('# Header', 'h1. Header'),
        ('`code`', '{{code}}'),
        ('```python\ncode\n```', '{code:python}\ncode\n{code}'),
        ('[link](url)', '[link|url]'),
        ('- item', '* item'),
        ('1. item', '# item'),
        ('~~strike~~', '-strike-'),
        ('> quote', 'bq. quote'),
    ]

    for markdown, expected in test_cases:
        result = markdown_to_jira_markup(markdown)
        assert (
            result == expected
        ), f'Failed for {repr(markdown)}: got {repr(result)}, expected {repr(expected)}'


def test_infer_repo_from_message():
    test_cases = [
        # Single GitHub URLs
        ('Clone https://github.com/demo123/demo1.git', ['demo123/demo1']),
        (
            'Check out https://github.com/OpenHands/OpenHands.git for details',
            ['OpenHands/OpenHands'],
        ),
        ('Visit https://github.com/microsoft/vscode', ['microsoft/vscode']),
        # Single GitLab URLs
        ('Deploy https://gitlab.com/demo1670324/demo1.git', ['demo1670324/demo1']),
        ('See https://gitlab.com/gitlab-org/gitlab', ['gitlab-org/gitlab']),
        (
            'Repository at https://gitlab.com/user.name/my-project.git',
            ['user.name/my-project'],
        ),
        # Single BitBucket URLs
        ('Pull from https://bitbucket.org/demo123/demo1.git', ['demo123/demo1']),
        (
            'Code is at https://bitbucket.org/atlassian/atlassian-connect-express',
            ['atlassian/atlassian-connect-express'],
        ),
        # Single direct owner/repo mentions
        ('Please deploy the OpenHands/OpenHands repo', ['OpenHands/OpenHands']),
        ('I need help with the microsoft/vscode repository', ['microsoft/vscode']),
        ('Check facebook/react for examples', ['facebook/react']),
        ('The torvalds/linux kernel', ['torvalds/linux']),
        # Multiple repositories in one message
        (
            'Compare https://github.com/user1/repo1.git with https://gitlab.com/user2/repo2',
            ['user1/repo1', 'user2/repo2'],
        ),
        (
            'Check facebook/react, microsoft/vscode, and google/angular',
            ['facebook/react', 'microsoft/vscode', 'google/angular'],
        ),
        (
            'URLs: https://github.com/python/cpython and https://bitbucket.org/atlassian/jira',
            ['python/cpython', 'atlassian/jira'],
        ),
        (
            'Mixed: https://github.com/owner/repo1.git and owner2/repo2 for testing',
            ['owner/repo1', 'owner2/repo2'],
        ),
        # Multi-line messages with multiple repos
        (
            'Please check these repositories:\n\nhttps://github.com/python/cpython\nhttps://gitlab.com/gitlab-org/gitlab\n\nfor updates',
            ['python/cpython', 'gitlab-org/gitlab'],
        ),
        (
            'I found issues in:\n- facebook/react\n- microsoft/vscode\n- google/angular',
            ['facebook/react', 'microsoft/vscode', 'google/angular'],
        ),
        # Duplicate handling (should not duplicate)
        ('Check https://github.com/user/repo.git and user/repo again', ['user/repo']),
        (
            'Both https://github.com/facebook/react and facebook/react library',
            ['facebook/react'],
        ),
        # URLs with parameters and fragments
        (
            'Clone https://github.com/user/repo.git?ref=main and https://gitlab.com/group/project.git#readme',
            ['user/repo', 'group/project'],
        ),
        # Complex mixed content (Git URLs have priority over direct mentions)
        (
            'Deploy https://github.com/main/app.git, check facebook/react docs, and https://bitbucket.org/team/utils',
            ['main/app', 'team/utils', 'facebook/react'],
        ),
        # Messages that should return empty list
        ('This is a message without a repo mention', []),
        ('Just some text about 12/25 date format', []),
        ('Version 1.0/2.0 comparison', []),
        ('http://example.com/not-a-git-url', []),
        ('Some/path/to/file.txt', []),
        ('Check the config.json file', []),
        # Edge cases with special characters
        ('https://github.com/My-User/My-Repo.git', ['My-User/My-Repo']),
        ('Check the my.user/my.repo repository', ['my.user/my.repo']),
        ('repos: user_1/repo-1 and user.2/repo_2', ['user_1/repo-1', 'user.2/repo_2']),
        # Large number of repositories
        ('Repos: a/b, c/d, e/f, g/h, i/j', ['a/b', 'c/d', 'e/f', 'g/h', 'i/j']),
        # Mixed with false positives that should be filtered
        ('Check user/repo and avoid 1.0/2.0 and file.txt', ['user/repo']),
    ]

    for message, expected in test_cases:
        result = infer_repo_from_message(message)
        assert (
            result == expected
        ), f'Failed for {repr(message)}: got {repr(result)}, expected {repr(expected)}'
