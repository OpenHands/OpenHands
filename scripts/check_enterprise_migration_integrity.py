#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSIONS_DIR = ROOT / 'enterprise' / 'migrations' / 'versions'
MIGRATION_FILENAME_RE = re.compile(r'^(?P<prefix>\d+)_.+\.py$')
MISSING = object()


def _literal_assignment(module: ast.Module, name: str) -> Any:
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return ast.literal_eval(node.value)

    return MISSING


def _format_paths(paths: list[Path]) -> str:
    return ', '.join(path.name for path in sorted(paths))


def _down_revisions(value: Any, path: Path, errors: list[str]) -> list[str]:
    if value is MISSING:
        errors.append(f'{path.name}: missing down_revision assignment')
        return []
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, str) for item in value):
            return list(value)

    errors.append(
        f'{path.name}: down_revision must be None, a string, or a sequence of strings'
    )
    return []


def check_migration_integrity(versions_dir: Path = DEFAULT_VERSIONS_DIR) -> list[str]:
    errors: list[str] = []
    migrations: list[dict[str, Any]] = []
    prefixes: defaultdict[str, list[Path]] = defaultdict(list)
    revisions: defaultdict[str, list[Path]] = defaultdict(list)
    referenced_revisions: set[str] = set()

    if not versions_dir.exists():
        return [f'Migration versions directory does not exist: {versions_dir}']

    for path in sorted(versions_dir.glob('*.py')):
        if path.name == '__init__.py':
            continue

        match = MIGRATION_FILENAME_RE.match(path.name)
        prefix = match.group('prefix') if match else None
        if prefix is None:
            errors.append(
                f'{path.name}: migration filename must start with a numeric prefix'
            )
        else:
            prefixes[prefix].append(path)

        try:
            module = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError as exc:
            errors.append(f'{path.name}: cannot parse migration: {exc}')
            continue

        revision = _literal_assignment(module, 'revision')
        if not isinstance(revision, str):
            errors.append(f'{path.name}: revision must be a string')
            continue

        revisions[revision].append(path)
        if prefix is not None and prefix != revision:
            errors.append(
                f'{path.name}: Filename prefix {prefix} does not match revision {revision}'
            )

        down_revision = _literal_assignment(module, 'down_revision')
        down_revisions = _down_revisions(down_revision, path, errors)
        referenced_revisions.update(down_revisions)
        migrations.append(
            {
                'path': path,
                'revision': revision,
                'down_revisions': down_revisions,
            }
        )

    for prefix, paths in sorted(prefixes.items()):
        if len(paths) > 1:
            errors.append(
                f'Duplicate migration filename prefix {prefix}: {_format_paths(paths)}'
            )

    for revision, paths in sorted(revisions.items()):
        if len(paths) > 1:
            errors.append(
                f'Duplicate migration revision {revision}: {_format_paths(paths)}'
            )

    known_revisions = set(revisions)
    for migration in migrations:
        for down_revision in migration['down_revisions']:
            if down_revision not in known_revisions:
                errors.append(
                    f'{migration["path"].name}: references missing down_revision '
                    f'{down_revision}'
                )

    heads = sorted(known_revisions - referenced_revisions)
    if migrations and len(heads) != 1:
        errors.append(
            f'Expected exactly one migration head, found {len(heads)}: '
            f'{", ".join(heads) if heads else "<none>"}'
        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check enterprise Alembic migration static integrity.'
    )
    parser.add_argument(
        '--versions-dir',
        type=Path,
        default=DEFAULT_VERSIONS_DIR,
        help='Path to the enterprise Alembic versions directory.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = check_migration_integrity(args.versions_dir)
    if errors:
        print('Enterprise migration integrity check failed:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 1

    print('Enterprise migration integrity checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
