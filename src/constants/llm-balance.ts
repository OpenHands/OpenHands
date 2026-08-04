/**
 * Agent-server endpoint reporting the LLM provider's credit balance.
 * Servers without balance support (or providers that can't report one)
 * answer 404 — callers must treat that as "hide the balance UI".
 */
export const LLM_BALANCE_PATH = "/api/llm/balance";
