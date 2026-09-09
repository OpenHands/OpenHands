import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import { buildHostDetectedConnection } from "#/utils/host-detected-provider-connections";
import { useAcpAuthStatus } from "./use-acp-auth-status";
import { useOpenAISubscriptionStatus } from "./use-llm-subscription-status";

/**
 * Provider rows for CLIs / ChatGPT that are already signed in on this
 * machine. These are not stored on `/api/llm/provider-connections`.
 */
export function useHostDetectedConnections(): {
  connections: ProviderConnection[];
  isChecking: boolean;
} {
  const { t } = useTranslation("openhands");
  const chatgpt = useOpenAISubscriptionStatus();
  const claude = useAcpAuthStatus("claude-code");
  const cursor = useAcpAuthStatus("cursor-cli");
  const opencode = useAcpAuthStatus("opencode");

  const connections = useMemo(() => {
    const rows: ProviderConnection[] = [];
    if (chatgpt.data?.connected) {
      rows.push(
        buildHostDetectedConnection(
          "chatgpt",
          t(I18nKey.SETTINGS$LLM_AUTH_TYPE_SUBSCRIPTION),
          "openai",
        ),
      );
    }
    if (claude.status === "authenticated") {
      rows.push(
        buildHostDetectedConnection(
          "anthropic",
          t(I18nKey.SETTINGS$LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION),
          "anthropic",
        ),
      );
    }
    if (cursor.status === "authenticated") {
      rows.push(
        buildHostDetectedConnection(
          "cursor-cli",
          mapProvider("cursor-cli"),
          "cursor-cli",
        ),
      );
    }
    if (opencode.status === "authenticated") {
      rows.push(
        buildHostDetectedConnection(
          "opencode",
          mapProvider("opencode"),
          "opencode",
        ),
      );
    }
    return rows;
  }, [
    chatgpt.data?.connected,
    claude.status,
    cursor.status,
    opencode.status,
    t,
  ]);

  return {
    connections,
    isChecking:
      (chatgpt.isFetching && chatgpt.data === undefined) ||
      claude.isChecking ||
      cursor.isChecking ||
      opencode.isChecking,
  };
}
