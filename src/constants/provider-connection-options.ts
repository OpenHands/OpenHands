/**
 * Providers that "Add provider" can actually configure with the fields we
 * collect (API key, optional base URL, ChatGPT device login, or a CLI
 * subscription such as Claude / Cursor / OpenCode).
 *
 * The agent-server's `/api/llm/providers` list is LiteLLM's full catalog —
 * TTS, embeddings, cloud IAM, Copilot, a `chatgpt` id that is not ChatGPT
 * Plus, etc. Dumping that into the modal made every option look like
 * "paste a key", which is wrong for most of them.
 */
export const PROVIDER_CONNECTION_AUTH_API_KEY = "api_key";
export const PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION =
  "openai_subscription";
export const PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION =
  "anthropic_subscription";
export const PROVIDER_CONNECTION_AUTH_CLI_SUBSCRIPTION = "cli_subscription";

export type ProviderConnectionAuth =
  | typeof PROVIDER_CONNECTION_AUTH_API_KEY
  | typeof PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION
  | typeof PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION
  | typeof PROVIDER_CONNECTION_AUTH_CLI_SUBSCRIPTION;

export type ProviderConnectionFieldRequirement =
  | "required"
  | "optional"
  | "hidden";

export interface ProviderConnectionOption {
  id: string;
  authModes: readonly ProviderConnectionAuth[];
  apiKey: ProviderConnectionFieldRequirement;
  baseUrl: ProviderConnectionFieldRequirement;
  showOpenHandsHelp?: boolean;
  /** Key for {@link AcpService.getAuthStatus} when this option uses a CLI login. */
  cliProbeKey?: string;
  /** Command the user runs in a terminal to sign in. */
  loginCommand?: string;
  /** Optional secret stored instead of (or as well as) a host CLI login. */
  oauthSecretName?: string;
}

export const OLLAMA_PROVIDER_ID = "ollama";
export const OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434";
export const OLLAMA_PLACEHOLDER_API_KEY = "ollama";
export const DEFAULT_OLLAMA_MODEL = "ollama/llama3.2";

const OPTIONS: readonly ProviderConnectionOption[] = [
  {
    id: "openai",
    authModes: [
      PROVIDER_CONNECTION_AUTH_API_KEY,
      PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION,
    ],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "anthropic",
    authModes: [
      PROVIDER_CONNECTION_AUTH_API_KEY,
      PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION,
    ],
    apiKey: "required",
    baseUrl: "optional",
    cliProbeKey: "claude-code",
    loginCommand: "claude auth login",
    oauthSecretName: "CLAUDE_CODE_OAUTH_TOKEN",
  },
  {
    id: "cursor-cli",
    authModes: [PROVIDER_CONNECTION_AUTH_CLI_SUBSCRIPTION],
    apiKey: "optional",
    baseUrl: "hidden",
    cliProbeKey: "cursor-cli",
    loginCommand: "agent login",
    oauthSecretName: "CURSOR_API_KEY",
  },
  {
    id: "opencode",
    authModes: [PROVIDER_CONNECTION_AUTH_CLI_SUBSCRIPTION],
    apiKey: "hidden",
    baseUrl: "hidden",
    cliProbeKey: "opencode",
    loginCommand: "opencode auth login",
  },
  {
    id: "openhands",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
    showOpenHandsHelp: true,
  },
  {
    id: "gemini",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "mistral",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "deepseek",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "moonshot",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "minimax",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "glm",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "qwen",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "groq",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "nvidia",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: "openrouter",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "optional",
  },
  {
    id: OLLAMA_PROVIDER_ID,
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "optional",
    baseUrl: "required",
  },
  {
    id: "custom",
    authModes: [PROVIDER_CONNECTION_AUTH_API_KEY],
    apiKey: "required",
    baseUrl: "required",
  },
];

const OPTIONS_BY_ID = new Map(OPTIONS.map((option) => [option.id, option]));

export function listProviderConnectionOptions(): readonly ProviderConnectionOption[] {
  return OPTIONS;
}

export function getProviderConnectionOption(
  id: string | null | undefined,
): ProviderConnectionOption | undefined {
  if (!id) return undefined;
  return OPTIONS_BY_ID.get(id);
}

export function isCliSubscriptionAuth(auth: ProviderConnectionAuth): boolean {
  return (
    auth === PROVIDER_CONNECTION_AUTH_CLI_SUBSCRIPTION ||
    auth === PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION
  );
}

export function resolveProviderConnectionFields(
  providerId: string | null | undefined,
  auth: ProviderConnectionAuth,
): {
  apiKey: ProviderConnectionFieldRequirement;
  baseUrl: ProviderConnectionFieldRequirement;
  showOpenHandsHelp: boolean;
} {
  const option = getProviderConnectionOption(providerId);
  if (
    auth === PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION ||
    auth === PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION
  ) {
    return { apiKey: "hidden", baseUrl: "hidden", showOpenHandsHelp: false };
  }
  return {
    apiKey: option?.apiKey ?? "required",
    baseUrl: option?.baseUrl ?? "optional",
    showOpenHandsHelp: Boolean(option?.showOpenHandsHelp),
  };
}
