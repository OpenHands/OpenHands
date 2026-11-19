import type { Provider } from "#/types/settings";

/** One entry of a git provider picker. */
export interface GitProviderOption {
  provider: Provider;
  /**
   * The `auth_providers.id` of the connection, or null when the entry stands
   * for a provider type and the instance picks its credentials.
   */
  gitProviderId: number | null;
  label: string;
}

export const gitProviderOptionKey = (option: GitProviderOption): string =>
  option.gitProviderId === null
    ? option.provider
    : `${option.provider}:${option.gitProviderId}`;

export const gitProviderOptionsMatch = (
  a: GitProviderOption | null | undefined,
  b: GitProviderOption | null | undefined,
): boolean =>
  !!a &&
  !!b &&
  a.provider === b.provider &&
  a.gitProviderId === b.gitProviderId;
