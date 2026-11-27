/**
 * Agent-server endpoint reporting the LLM provider's credit balance.
 * Servers without balance support (or providers that can't report one)
 * answer 404 — callers must treat that as "hide the balance UI".
 */
export const LLM_BALANCE_PATH = "/api/llm/balance";

/**
 * Ceiling on a single balance request. A server that accepts the connection
 * and then never answers would otherwise leave the request in flight forever:
 * the query sets `retry: false`, which suppresses retries but does not bound a
 * stuck request, so the balance card would stay hidden with nothing surfaced.
 */
export const LLM_BALANCE_TIMEOUT_MS = 10_000;
