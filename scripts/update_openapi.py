#!/usr/bin/env python3
"""Update OpenHands OpenAPI documentation.

Generates the OpenAPI specification from the FastAPI application and writes it
to docs/openapi.json.

Usage:
    python scripts/update_openapi.py

Behavior:
- Uses openhands.server.app.app.openapi() to build the spec.
- Preserves existing "servers" from docs/openapi.json if present; otherwise
  writes sensible defaults.
- Sets info.version to openhands.__version__.
- Sanitizes endpoint descriptions to remove code blocks and internal-only sections.
- Excludes operational/UI-only convenience endpoints:
  - /server_info
  - /api/conversations/{conversation_id}/vscode-url
  - /api/conversations/{conversation_id}/web-hosts
- Creates a backup docs/openapi.json.backup before overwriting.

Output:
- Prints OpenAPI and API versions, endpoint count, servers count, and sample endpoints.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings and logs during import
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
os.environ['OPENHANDS_LOG_LEVEL'] = 'CRITICAL'

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from openhands import __version__
    from openhands.server.app import app
except ImportError as e:
    print(f'Error importing OpenHands modules: {e}')
    print(
        "Make sure you're running this script from the project root and dependencies are installed."
    )
    sys.exit(1)


def _sanitize_description(text: str) -> str:
    """Remove internal, code-centric, or redundant sections from endpoint descriptions.

    - Strip fenced code blocks
    - Remove Args/Returns/Raises/Example/Examples/Notes sections
    - Remove inline curl examples
    - Avoid provider-implementation specifics like LiteLLM/Bedrock
    """
    import re

    if not text:
        return text

    # Remove fenced code blocks
    text = re.sub(r'```[\s\S]*?```', '', text, flags=re.MULTILINE)

    # Remove common docstring sections (until next blank line or end)
    for header in [
        r'Args?:',
        r'Returns?:',
        r'Raises?:',
        r'Example[s]?:',
        r'Notes?:',
    ]:
        text = re.sub(rf'(?ms)^\s*{header}.*?(?:\n\s*\n|\Z)', '', text)

    # Remove lines that contain curl examples
    text = re.sub(r'(?im)^.*\bcurl\b.*$', '', text)

    # Generalize provider-implementation specifics
    text = re.sub(r'\bLiteLLM\b', 'configured model providers', text)
    text = re.sub(r'\blitellm\b', 'configured providers', text)
    text = re.sub(r'\bBedrock\b', '', text)

    # Collapse excessive blank lines and trim
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _sanitize_spec(spec: dict) -> dict:
    """Sanitize descriptions and summaries to be public-API friendly."""
    path_summary_overrides = {
        '/api/options/models': 'List Supported Models',
        '/api/options/agents': 'List Agents',
        '/api/options/security-analyzers': 'List Security Analyzers',
        '/api/conversations/{conversation_id}/list-files': 'List Workspace Files',
        '/api/conversations/{conversation_id}/select-file': 'Get File Content',
        '/api/conversations/{conversation_id}/zip-directory': 'Download Workspace Archive',
    }
    path_description_overrides = {
        '/api/options/models': 'List model identifiers available on this server based on configured providers.',
        '/api/options/agents': 'List available agent types supported by this server.',
        '/api/options/security-analyzers': 'List supported security analyzers.',
        '/api/conversations/{conversation_id}/list-files': 'List workspace files visible to the conversation runtime. Applies .gitignore and internal ignore rules.',
        '/api/conversations/{conversation_id}/select-file': 'Return the content of the given file from the conversation workspace.',
        '/api/conversations/{conversation_id}/zip-directory': 'Return a ZIP archive of the current conversation workspace.',
    }

    for path, methods in list(spec.get('paths', {}).items()):
        for method, meta in list(methods.items()):
            if not isinstance(meta, dict):
                continue
            # Override overly specific summaries where helpful
            if path in path_summary_overrides:
                meta['summary'] = path_summary_overrides[path]
            # Override description if provided; otherwise sanitize
            if path in path_description_overrides:
                meta['description'] = path_description_overrides[path]
            elif 'description' in meta and isinstance(meta['description'], str):
                meta['description'] = _sanitize_description(meta['description'])

    return spec


def generate_openapi_spec():
    """Generate the OpenAPI specification from the FastAPI app."""
    spec = app.openapi()

    # Explicitly exclude certain endpoints that are operational, experimental, or UI-only convenience
    excluded_endpoints = [
        '/api/conversations/{conversation_id}/exp-config',  # Internal experimentation endpoint
        '/server_info',  # Operational/system diagnostics
        '/api/conversations/{conversation_id}/vscode-url',  # UI/runtime convenience
        '/api/conversations/{conversation_id}/web-hosts',  # UI/runtime convenience
    ]

    if 'paths' in spec:
        for endpoint in excluded_endpoints:
            if endpoint in spec['paths']:
                del spec['paths'][endpoint]
                print(f'Excluded endpoint: {endpoint}')

    # Sanitize descriptions and summaries
    spec = _sanitize_spec(spec)

    return spec


def load_current_spec(spec_path):
    """Load the current OpenAPI specification if it exists."""
    if spec_path.exists():
        with open(spec_path, 'r') as f:
            return json.load(f)
    return {}


def update_openapi_spec(spec_path, backup=True):
    """Update the OpenAPI specification file."""
    # Generate new spec
    new_spec = generate_openapi_spec()

    # Load current spec for server information
    current_spec = load_current_spec(spec_path)

    # Preserve server information from current spec if it exists
    if 'servers' in current_spec:
        new_spec['servers'] = current_spec['servers']
    else:
        # Default servers if none exist
        new_spec['servers'] = [
            {'url': 'https://app.all-hands.dev', 'description': 'Production server'},
            {'url': 'http://localhost:3000', 'description': 'Local server'},
        ]

    # Update version to match the package version
    new_spec['info']['version'] = __version__

    # Backup current file if requested
    if backup and spec_path.exists():
        backup_path = spec_path.with_suffix('.json.backup')
        spec_path.rename(backup_path)
        print(f'Backed up current spec to {backup_path}')

    # Write new spec
    with open(spec_path, 'w') as f:
        json.dump(new_spec, f, indent=2)

    return new_spec


def main():
    """Main function."""
    spec_path = project_root / 'docs' / 'openapi.json'

    print('Updating OpenAPI specification...')
    print(f'Target file: {spec_path}')

    try:
        new_spec = update_openapi_spec(spec_path)

        print('✅ Successfully updated OpenAPI specification!')
        print(f'   OpenAPI version: {new_spec.get("openapi", "N/A")}')
        print(f'   API version: {new_spec.get("info", {}).get("version", "N/A")}')
        print(f'   Total endpoints: {len(new_spec.get("paths", {}))}')
        print(f'   Servers: {len(new_spec.get("servers", []))}')

        # List some key endpoints
        paths = list(new_spec.get('paths', {}).keys())
        if paths:
            print('   Sample endpoints:')
            for path in sorted(paths)[:5]:
                methods = list(new_spec['paths'][path].keys())
                print(f'     {path}: {methods}')
            if len(paths) > 5:
                print(f'     ... and {len(paths) - 5} more')

    except Exception as e:
        print(f'❌ Error updating OpenAPI specification: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
