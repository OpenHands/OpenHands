import pytest

from openhands.utils.git import ensure_valid_git_branch_name, is_valid_git_branch_name


def test_is_valid_git_branch_name_accepts_common_hosted_git_branch_names():
    for branch_name in (
        'main',
        'feature/test-branch',
        'release/1.2.3',
        'dependabot/npm_and_yarn/sdk-1.2.3',
        'renovate/grouped-updates',
    ):
        assert is_valid_git_branch_name(branch_name) is True


@pytest.mark.parametrize(
    'branch_name',
    [
        'main; git -C /workspace/TylersTestRepo remote -v >/root/file.txt;',
        'feature branch',
        'feature..branch',
        '-branch',
    ],
)
def test_ensure_valid_git_branch_name_rejects_invalid_git_syntax(branch_name):
    with pytest.raises(ValueError, match='Common GitHub/GitLab/Bitbucket branch names'):
        ensure_valid_git_branch_name(branch_name)
