import React from "react";
import { convertRawProvidersToList } from "#/utils/convert-raw-providers-to-list";
import { useSettings } from "./query/use-settings";

export const useUserProviders = () => {
  const { data: settings, isLoading: isLoadingSettings } = useSettings();

  const providers = React.useMemo(
    () => convertRawProvidersToList(settings?.provider_tokens_set),
    [settings?.provider_tokens_set],
  );

  return {
    providers,
    isLoadingSettings,
  };
};
