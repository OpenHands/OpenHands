import { I18nKey } from "#/i18n/declaration";

/**
 * A credential an ACP provider authenticates with. The {@link name} is both
 * the global-secret name and the env var the agent-server injects into the
 * ACP subprocess — keeping them identical is what makes a saved secret
 * actually reach the CLI.
 */
export interface ACPProviderSecretField {
  name: string;
  secret?: boolean;
  multiline?: boolean;
  hint_key: I18nKey;
  hint_values?: Record<string, string>;
}

const ACP_PROVIDER_SECRETS: Record<string, ACPProviderSecretField[]> = {
  "claude-code": [
    {
      name: "CLAUDE_CODE_OAUTH_TOKEN",
      secret: true,
      hint_key: I18nKey.SETTINGS$ACP_SECRET_OAUTH_TOKEN_HINT,
    },
    {
      name: "ANTHROPIC_API_KEY",
      secret: true,
      hint_key: I18nKey.SETTINGS$ACP_SECRET_API_KEY_HINT,
    },
    {
      name: "ANTHROPIC_BASE_URL",
      hint_key: I18nKey.SETTINGS$ACP_SECRET_BASE_URL_HINT,
    },
  ],
  codex: [
    {
      name: "CODEX_AUTH_JSON",
      secret: true,
      multiline: true,
      hint_key: I18nKey.SETTINGS$ACP_SECRET_FILE_BLOB_HINT,
      hint_values: { file: "~/.codex/auth.json" },
    },
    {
      name: "OPENAI_API_KEY",
      secret: true,
      hint_key: I18nKey.SETTINGS$ACP_SECRET_API_KEY_HINT,
    },
    {
      name: "OPENAI_BASE_URL",
      hint_key: I18nKey.SETTINGS$ACP_SECRET_BASE_URL_HINT,
    },
  ],
  "gemini-cli": [
    {
      name: "GOOGLE_APPLICATION_CREDENTIALS_JSON",
      secret: true,
      multiline: true,
      hint_key: I18nKey.SETTINGS$ACP_SECRET_FILE_BLOB_HINT,
      hint_values: {
        file: "~/.config/gcloud/application_default_credentials.json",
      },
    },
    {
      name: "GOOGLE_CLOUD_PROJECT",
      hint_key: I18nKey.SETTINGS$ACP_SECRET_GCP_PROJECT_HINT,
    },
    {
      name: "GOOGLE_CLOUD_LOCATION",
      hint_key: I18nKey.SETTINGS$ACP_SECRET_GCP_LOCATION_HINT,
    },
    {
      name: "GOOGLE_GENAI_USE_VERTEXAI",
      hint_key: I18nKey.SETTINGS$ACP_SECRET_VERTEXAI_FLAG_HINT,
    },
  ],
};

/**
 * Returns credential fields for the given ACP provider key.
 * Returns [] for custom presets and unknown keys.
 */
export function getAcpProviderSecrets(
  key: string | null | undefined,
): ACPProviderSecretField[] {
  if (!key) return [];
  return ACP_PROVIDER_SECRETS[key] ?? [];
}

/**
 * Returns [credential, conflicting] pairs where both are set (typed or saved).
 * Claude's OAuth token authenticates against Anthropic directly; an
 * ANTHROPIC_BASE_URL set alongside it silently routes requests elsewhere
 * and breaks the token's bearer auth.
 */
const ACP_CREDENTIAL_CONFLICTS: Record<string, [string, string][]> = {
  "claude-code": [["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"]],
};

export function getAcpCredentialConflicts(
  key: string | null | undefined,
  hasValueFor: (name: string) => boolean,
): Array<[string, string]> {
  if (!key) return [];
  const pairs = ACP_CREDENTIAL_CONFLICTS[key] ?? [];
  return pairs.filter(([a, b]) => hasValueFor(a) && hasValueFor(b));
}
