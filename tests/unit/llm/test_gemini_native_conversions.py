from openhands.llm.gemini_native_conversions import (
    openai_tool_to_gemini_function_declaration,
)


def test_openai_tool_to_gemini_function_declaration_minimal():
    tool = {
        "type": "function",
        "function": {
            "name": "mouse_click",
            "description": "Click at normalized coords",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string"},
                },
                "required": ["x", "y", "button"],
            },
        },
    }

    decl = openai_tool_to_gemini_function_declaration(tool)
    assert decl["name"] == "mouse_click"
    assert "parameters" in decl
    assert decl["parameters"]["type"] == "object"
    assert "x" in decl["parameters"]["properties"]


