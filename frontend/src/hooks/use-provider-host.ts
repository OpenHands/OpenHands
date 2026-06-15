import { useConfig } from "#/hooks/query/use-config";
import { useSettings } from "#/hooks/query/use-settings";
import { Provider } from "#/types/settings";

/**
 * Resolve the effective host for a git provider: a per-token override from
 * user settings if present, otherwise the deployment-wide default from the
 * web client config, otherwise null.
 */
export const useProviderHost = (
  provider: Provider | null | undefined,
): string | null => {
  const { data: settings } = useSettings();
  const { data: config } = useConfig();
  if (!provider) return null;
  return (
    settings?.provider_tokens_set[provider] ||
    config?.provider_default_hosts?.[provider] ||
    null
  );
};
