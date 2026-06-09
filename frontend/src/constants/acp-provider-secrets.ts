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
      name: "OPENAI_API_KEY",
      secret: true,
      hint_key: I18nKey.SETTINGS$ACP_SECRET_API_KEY_HINT,
    },
    {
      name: "OPENAI_BASE_URL",
      hint_key: I18nKey.SETTINGS$ACP_SECRET_BASE_URL_HINT,
    },
  ],
};

/**
 * Returns credential fields for the given ACP provider key.
 * Returns [] for providers with no API-key credential (Gemini CLI),
 * custom presets, and unknown keys.
 */
export function getAcpProviderSecrets(
  key: string | null | undefined,
): ACPProviderSecretField[] {
  if (!key) return [];
  return ACP_PROVIDER_SECRETS[key] ?? [];
}

/**
 * Returns [credential, conflicting] pairs where both are set (typed or saved).
 * Used to warn users about conflicting credential combinations.
 */
const ACP_CREDENTIAL_CONFLICTS: Record<string, [string, string][]> = {
  "claude-code": [["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]],
};

export function getAcpCredentialConflicts(
  key: string | null | undefined,
  hasValueFor: (name: string) => boolean,
): Array<[string, string]> {
  if (!key) return [];
  const pairs = ACP_CREDENTIAL_CONFLICTS[key] ?? [];
  return pairs.filter(([a, b]) => hasValueFor(a) && hasValueFor(b));
}
