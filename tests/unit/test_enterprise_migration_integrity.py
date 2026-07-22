from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = 'check_enterprise_migration_integrity'
_saved_spec = None


def load_module():
    global _saved_spec
    if _saved_spec is not None:
        module = importlib.util.module_from_spec(_saved_spec)
        spec_loader = _saved_spec.loader
        spec_loader.exec_module(module)
        sys.modules[MODULE_NAME] = module
        return module
    path = ROOT / 'scripts' / 'check_enterprise_migration_integrity.py'
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f'Unable to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _saved_spec = spec
    sys.modules[MODULE_NAME] = module
    return module


def write_migration(
    versions_dir: Path,
    filename: str,
    *,
    revision: str,
    down_revision: str | None,
) -> None:
    down_revision_value = repr(down_revision) if down_revision is not None else 'None'
    (versions_dir / filename).write_text(
        '\n'.join(
            [
                '"""Test migration."""',
                '',
                f'revision = {revision!r}',
                f'down_revision = {down_revision_value}',
                'branch_labels = None',
                'depends_on = None',
                '',
                'def upgrade():',
                '    pass',
                '',
                'def downgrade():',
                '    pass',
                '',
            ]
        )
    )


@pytest.fixture
def versions_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'versions'
    path.mkdir()
    return path


def test_valid_linear_migrations_pass(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='001',
        down_revision=None,
    )
    write_migration(
        versions_dir,
        '002_add_user_email.py',
        revision='002',
        down_revision='001',
    )

    assert module.check_migration_integrity(versions_dir) == []


def test_duplicate_filename_prefix_fails(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='001',
        down_revision=None,
    )
    write_migration(
        versions_dir,
        '001_add_user_email.py',
        revision='002',
        down_revision='001',
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any('Duplicate migration filename prefix 001' in error for error in errors)


def test_duplicate_revision_fails(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='001',
        down_revision=None,
    )
    write_migration(
        versions_dir,
        '002_add_user_email.py',
        revision='001',
        down_revision='001',
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any('Duplicate migration revision 001' in error for error in errors)


def test_filename_prefix_must_match_revision(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='002',
        down_revision=None,
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any(
        'Filename prefix 001 does not match revision 002' in error for error in errors
    )


def test_missing_down_revision_fails(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='001',
        down_revision=None,
    )
    write_migration(
        versions_dir,
        '002_add_user_email.py',
        revision='002',
        down_revision='999',
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any('references missing down_revision 999' in error for error in errors)


def test_missing_down_revision_assignment_fails(versions_dir: Path):
    """Test when down_revision is completely absent from the file."""
    module = load_module()
    (versions_dir / '001_create_users.py').write_text(
        '\n'.join(
            [
                '"""Test migration."""',
                '',
                "revision = '001'",
                'branch_labels = None',
                'depends_on = None',
                '',
                'def upgrade():',
                '    pass',
                '',
                'def downgrade():',
                '    pass',
                '',
            ]
        )
    )
    write_migration(
        versions_dir,
        '002_add_user_email.py',
        revision='002',
        down_revision='001',
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any('missing down_revision assignment' in error for error in errors)


def test_multiple_heads_fail(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='001',
        down_revision=None,
    )
    write_migration(
        versions_dir,
        '002_add_user_email.py',
        revision='002',
        down_revision='001',
    )
    write_migration(
        versions_dir,
        '003_add_user_name.py',
        revision='003',
        down_revision='001',
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any('Expected exactly one migration head' in error for error in errors)


def test_revision_must_be_string(versions_dir: Path):
    module = load_module()
    (versions_dir / '001_create_users.py').write_text(
        '\n'.join(
            [
                '"""Test migration with non-string revision."""',
                '',
                'revision = 123',
                'down_revision = None',
                'branch_labels = None',
                'depends_on = None',
                '',
                'def upgrade():',
                '    pass',
                '',
                'def downgrade():',
                '    pass',
                '',
            ]
        )
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any(
        '001_create_users.py: revision must be a string' in error for error in errors
    )


def test_invalid_down_revision_type_fails(versions_dir: Path):
    module = load_module()
    write_migration(
        versions_dir,
        '001_create_users.py',
        revision='001',
        down_revision=None,
    )
    # Write a migration with an invalid down_revision type (integer instead of string)
    (versions_dir / '002_add_user_email.py').write_text(
        '\n'.join(
            [
                '"""Test migration."""',
                '',
                "revision = '002'",
                'down_revision = 123',  # invalid type: must be None, string, or sequence
                'branch_labels = None',
                'depends_on = None',
                '',
                'def upgrade():',
                '    pass',
                '',
                'def downgrade():',
                '    pass',
                '',
            ]
        )
    )

    errors = module.check_migration_integrity(versions_dir)

    assert any(
        'down_revision must be None, a string, or a sequence of strings' in error
        for error in errors
    )
