import { useQuery } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";
import { SettingsSchema } from "#/types/settings";
import { useIsAuthed } from "./use-is-authed";

const useSettingsSchema = (
  type: "agent" | "conversation",
  fallbackSchema?: SettingsSchema | null,
) => {
  const isOnIntermediatePage = useIsOnIntermediatePage();
  const { data: userIsAuthenticated } = useIsAuthed();
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["settings-schema", type],
    queryFn:
      type === "conversation"
        ? SettingsService.getConversationSettingsSchema
        : SettingsService.getSettingsSchema,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    enabled: !fallbackSchema && !isOnIntermediatePage && !!userIsAuthenticated,
    meta: {
      disableToast: true,
    },
  });

  if (fallbackSchema) {
    return {
      data: fallbackSchema,
      isLoading: false,
      isFetching: false,
    };
  }

  return {
    data,
    isLoading,
    isFetching,
  };
};

export const useAgentSettingsSchema = (
  fallbackSchema?: SettingsSchema | null,
) => useSettingsSchema("agent", fallbackSchema);

export const useConversationSettingsSchema = (
  fallbackSchema?: SettingsSchema | null,
) => useSettingsSchema("conversation", fallbackSchema);
