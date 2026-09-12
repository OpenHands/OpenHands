import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { supportsConversationRuntimeRoutes } from "#/api/agent-server-client-options";
import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getConversationServerInfo,
  usesIsolatedWorkspace,
} from "#/api/conversation-workspace";

export function useConversationWorkspace() {
  const { t } = useTranslation();
  const { backend } = useActiveBackend();
  const query = useQuery({
    queryKey: [
      "conversation-workspace",
      backend.id,
      backend.host,
      backend.connectionRevision,
    ],
    queryFn: () => getConversationServerInfo(),
    enabled: backend.kind === "local" && !!backend.host,
    staleTime: 60_000,
    retry: false,
    meta: { disableToast: true },
  });
  const isolated =
    backend.kind === "local" && usesIsolatedWorkspace(query.data);
  const clientUnsupported = isolated && !supportsConversationRuntimeRoutes();
  return {
    clientUnsupported,
    clientUnsupportedMessage: clientUnsupported
      ? t(I18nKey.HOME$ISOLATED_WORKSPACE_UPGRADE)
      : null,
    isolated,
    unsupportedMessage: isolated
      ? t(I18nKey.HOME$ISOLATED_WORKSPACE_NOTICE)
      : null,
  };
}
