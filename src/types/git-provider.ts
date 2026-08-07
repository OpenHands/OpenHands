/**
 * Non-secret Git provider metadata stored in
 * `misc_settings.app_preferences.git_providers`. Credentials live only in
 * SecretsService under conventional `GIT_PROVIDER_{id}_*` names.
 */
export type GitProviderAuthMethod = "pat" | "password" | "ssh";

export type GitProviderPreference = {
  id: string;
  label: string;
  host: string;
  auth_method: GitProviderAuthMethod;
};

/** Provider ids must fit secret-name limits (letters/numbers/underscores, ≤64). */
export const GIT_PROVIDER_ID_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;

export const GIT_PROVIDER_AUTH_METHODS: GitProviderAuthMethod[] = [
  "pat",
  "password",
  "ssh",
];
