"""Dependency-free conversions for Gemini-native tool loops.

This module intentionally avoids importing google.genai so unit tests can run
without optional dependencies.

Keep this file lean: only keep helpers used by production code.
"""

from __future__ import annotations

from typing import Any

def openai_tool_to_gemini_function_declaration(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style tool dict to a Gemini FunctionDeclaration-like dict.

    Expected OpenAI-style tool shape:
        {"type": "function", "function": {"name": str, "description": str, "parameters": dict}}
    """
    fn = tool.get('function') if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        raise ValueError('Invalid tool: missing function dict')

    name = fn.get('name')
    if not isinstance(name, str) or not name:
        raise ValueError('Invalid tool: missing function.name')

    description = fn.get('description') or ''
    if not isinstance(description, str):
        description = str(description)

    parameters = fn.get('parameters') or {'type': 'object', 'properties': {}}
    if not isinstance(parameters, dict):
        raise ValueError('Invalid tool: function.parameters must be a dict')

    # Sanitize parameters for Gemini compatibility (e.g., string enums)
    parameters = _sanitize_parameters(parameters)

    # Gemini expects JSON-schema-like parameters
    return {
        'name': name,
        'description': description,
        'parameters': parameters,
    }


def _sanitize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize parameters for Gemini compatibility.

    Gemini's SDK (google-genai) is strict about types:
    - Enums must be strings.
    - If enum is used, type should be string.
    """
    if not isinstance(params, dict):
        return params

    new_params = params.copy()

    # Handle enum: convert all values to strings and force type to string
    if 'enum' in new_params and isinstance(new_params['enum'], list):
        new_params['enum'] = [str(v) for v in new_params['enum']]
        # If we have an enum, Gemini generally expects it to be a string type
        new_params['type'] = 'string'

    # Remove additionalProperties if present (Gemini API often rejects it)
    new_params.pop('additionalProperties', None)
    new_params.pop('additional_properties', None)

    # Handle union types (e.g. type=['string', 'number']) -> force to 'string'
    if 'type' in new_params and isinstance(new_params['type'], list):
        # Gemini doesn't support list of types. Fallback to string as catch-all.
        new_params['type'] = 'string'

    # Recursively handle properties
    if 'properties' in new_params and isinstance(new_params['properties'], dict):
        new_props = {}
        for k, v in new_params['properties'].items():
            if isinstance(v, dict):
                new_props[k] = _sanitize_parameters(v)
            else:
                new_props[k] = v
        new_params['properties'] = new_props

    # Recursively handle items (for arrays)
    if 'items' in new_params and isinstance(new_params['items'], dict):
        new_params['items'] = _sanitize_parameters(new_params['items'])

    return new_params


