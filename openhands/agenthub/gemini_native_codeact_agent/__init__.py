from openhands.agenthub.gemini_native_codeact_agent.gemini_native_codeact_agent import (
    GeminiNativeCodeActAgent,
)
from openhands.controller.agent import Agent

Agent.register('GeminiNativeCodeActAgent', GeminiNativeCodeActAgent)


