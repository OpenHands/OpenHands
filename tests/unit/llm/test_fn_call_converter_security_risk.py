from openhands.sdk.llm.mixins.fn_call_converter import (
    convert_non_fncall_messages_to_fncall_messages,
)


def test_execute_bash_alias_does_not_require_security_risk():
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'terminal',
                'description': 'Execute a shell command.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {'type': 'string'},
                        'security_risk': {
                            'type': 'string',
                            'enum': ['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN'],
                        },
                    },
                    'required': ['command', 'security_risk'],
                },
            },
        }
    ]
    messages = [
        {'role': 'user', 'content': 'List the current directory.'},
        {
            'role': 'assistant',
            'content': '<function=execute_bash>\n'
            '<parameter=command>ls</parameter>\n'
            '</function>',
        },
    ]

    converted = convert_non_fncall_messages_to_fncall_messages(messages, tools)

    tool_call = converted[1]['tool_calls'][0]
    assert tool_call['function']['name'] == 'terminal'
    assert tool_call['function']['arguments'] == '{"command": "ls"}'
