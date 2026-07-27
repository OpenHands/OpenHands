import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { isNoBackend } from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SettingsSchema } from "#/types/settings";
import { withLlmSubscriptionSchemaFields } from "#/utils/llm-subscription-schema";
import { useIsAuthed } from "./use-is-authed";

const useSettingsSchema = (
  type: "agent" | "conversation",
  fallbackSchema?: SettingsSchema | null,
) => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const { backend, orgId } = useActiveBackend();
  const hasBackend = !isNoBackend(backend);
  const { data, error, isLoading, isFetching } = useQuery({
    queryKey: [
      "settings-schema",
      type,
      backend.id,
      orgId,
      backend.kind,
      backend.host,
      backend.apiKey,
    ],
    queryFn:
      type === "conversation"
        ? SettingsService.getConversationSettingsSchema
        : SettingsService.getSettingsSchema,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    enabled: !fallbackSchema && !!userIsAuthenticated && hasBackend,
    meta: {
      disableToast: true,
    },
  });

  const fallbackData = useMemo(
    () =>
      type === "agent"
        ? withLlmSubscriptionSchemaFields(fallbackSchema)
        : fallbackSchema,
    [fallbackSchema, type],
  );

  const queryData = useMemo(
    () => (type === "agent" ? withLlmSubscriptionSchemaFields(data) : data),
    [data, type],
  );

  if (fallbackSchema) {
    return {
      data: fallbackData,
      error: null,
      isLoading: false,
      isFetching: false,
    };
  }

  return {
    data: queryData,
    error,
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
