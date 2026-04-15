from importlib import import_module

import pytest


PUBLIC_IMPORT_CASES = [
    (
        'openhands.sdk',
        'ConversationStats',
        'openhands.sdk.conversation.conversation_stats',
        'ConversationStats',
    ),
    (
        'openhands.sdk',
        'LLMSummarizingCondenser',
        'openhands.sdk.context.condenser',
        'LLMSummarizingCondenser',
    ),
    (
        'openhands.sdk.conversation',
        'ConversationExecutionStatus',
        'openhands.sdk.conversation.state',
        'ConversationExecutionStatus',
    ),
    (
        'openhands.sdk.context',
        'AgentContext',
        'openhands.sdk.context.agent_context',
        'AgentContext',
    ),
    (
        'openhands.sdk.context.skills',
        'KeywordTrigger',
        'openhands.sdk.context.skills.trigger',
        'KeywordTrigger',
    ),
    (
        'openhands.sdk.context.skills',
        'TaskTrigger',
        'openhands.sdk.context.skills.trigger',
        'TaskTrigger',
    ),
    ('openhands.sdk.event', 'EventID', 'openhands.sdk.event.types', 'EventID'),
    ('openhands.sdk.llm', 'Metrics', 'openhands.sdk.llm.utils.metrics', 'Metrics'),
    (
        'openhands.sdk.llm',
        'TokenUsage',
        'openhands.sdk.llm.utils.metrics',
        'TokenUsage',
    ),
    (
        'openhands.sdk.security',
        'SecurityAnalyzerBase',
        'openhands.sdk.security.analyzer',
        'SecurityAnalyzerBase',
    ),
    (
        'openhands.sdk.security',
        'LLMSecurityAnalyzer',
        'openhands.sdk.security.llm_analyzer',
        'LLMSecurityAnalyzer',
    ),
    (
        'openhands.sdk.security',
        'ConfirmationPolicyBase',
        'openhands.sdk.security.confirmation_policy',
        'ConfirmationPolicyBase',
    ),
    (
        'openhands.sdk.security',
        'AlwaysConfirm',
        'openhands.sdk.security.confirmation_policy',
        'AlwaysConfirm',
    ),
    (
        'openhands.sdk.security',
        'NeverConfirm',
        'openhands.sdk.security.confirmation_policy',
        'NeverConfirm',
    ),
    (
        'openhands.sdk.security',
        'ConfirmRisky',
        'openhands.sdk.security.confirmation_policy',
        'ConfirmRisky',
    ),
    (
        'openhands.sdk.workspace',
        'FileOperationResult',
        'openhands.sdk.workspace.models',
        'FileOperationResult',
    ),
]


@pytest.mark.parametrize(
    ('public_module', 'public_name', 'deep_module', 'deep_name'),
    PUBLIC_IMPORT_CASES,
)
def test_sdk_public_import_resolves_to_same_symbol(
    public_module: str,
    public_name: str,
    deep_module: str,
    deep_name: str,
) -> None:
    public_symbol = getattr(import_module(public_module), public_name)
    deep_symbol = getattr(import_module(deep_module), deep_name)

    assert public_symbol is deep_symbol
