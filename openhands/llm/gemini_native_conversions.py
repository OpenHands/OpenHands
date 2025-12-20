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

    # Gemini expects JSON-schema-like parameters
    return {
        'name': name,
        'description': description,
        'parameters': parameters,
    }


