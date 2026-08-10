/**
 * Path constants for the agent-server Provider Connection endpoints.
 *
 * Kept separate from the service so the service file never has a literal
 * "/api/..." string adjacent to its HTTP call (see the
 * no-direct-agent-server-calls guard), mirroring the LLM subscription
 * service pattern.
 */
export const PROVIDER_CONNECTIONS_PATH = "/api/llm/connections";
export const PROVIDER_CONNECTION_PATH = (id: string) =>
  `${PROVIDER_CONNECTIONS_PATH}/${encodeURIComponent(id)}`;
export const PROVIDER_CONNECTION_VALIDATE_PATH = (id: string) =>
  `${PROVIDER_CONNECTION_PATH(id)}/validate`;
