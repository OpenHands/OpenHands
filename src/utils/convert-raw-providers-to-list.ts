import { Provider, ProviderOptions } from "#/types/settings";

// `provider_tokens_set` can carry non-git entries (e.g. `enterprise_sso`
// from an SSO login); only keep the git providers the UI knows about.
const isProvider = (key: string): key is Provider =>
  Object.hasOwn(ProviderOptions, key);

export const convertRawProvidersToList = (
  raw: Partial<Record<Provider, string | null>> | undefined,
): Provider[] => {
  if (!raw) return [];
  const list: Provider[] = [];
  for (const key of Object.keys(raw)) {
    if (isProvider(key)) {
      list.push(key);
    }
  }
  return list;
};
