import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Link } from "react-router";
import { X } from "lucide-react";
import { FaTriangleExclamation } from "react-icons/fa6";
import { I18nKey } from "#/i18n/declaration";
import { useMcpConversationHealth } from "#/hooks/query/use-mcp-conversation-health";
import { useMcpWarningDismissStore } from "#/stores/mcp-warning-dismiss-store";
import { getMcpFailureLabel } from "#/utils/mcp-failure-i18n";
import { cn } from "#/utils/utils";

interface McpUnavailableBannerProps {
  conversationId: string | null | undefined;
}

export function McpUnavailableBanner({
  conversationId,
}: McpUnavailableBannerProps) {
  const { t } = useTranslation();
  const { unhealthyServers, isLoading } =
    useMcpConversationHealth(!!conversationId);
  const dismissedKeys = useMcpWarningDismissStore(
    (state) => state.dismissedKeys,
  );
  const dismiss = useMcpWarningDismissStore((state) => state.dismiss);

  if (!conversationId || isLoading) {
    return null;
  }

  const isServerDismissed = (serverId: string) =>
    dismissedKeys.includes(`${conversationId}:${serverId}`);

  const visibleServers = unhealthyServers.filter(
    ({ serverId }) => !isServerDismissed(serverId),
  );

  if (visibleServers.length === 0) {
    return null;
  }

  const dismissAll = () => {
    visibleServers.forEach(({ serverId }) => dismiss(conversationId, serverId));
  };

  return (
    <div
      className={cn(
        "w-full rounded-lg p-3 border border-amber-500/60 bg-amber-500/10",
        "flex gap-3 items-start text-white",
      )}
      data-testid="mcp-unavailable-banner"
    >
      <FaTriangleExclamation
        className="text-amber-400 shrink-0 mt-0.5"
        aria-hidden
      />
      <div className="min-w-0 flex-1 flex flex-col gap-1">
        {visibleServers.length === 1 ? (
          <p className="text-sm" data-testid="mcp-unavailable-banner-content">
            <Trans
              i18nKey={I18nKey.CONVERSATION$MCP_UNAVAILABLE_SINGLE}
              values={{
                serverName: visibleServers[0].serverId,
                detail: getMcpFailureLabel(
                  t,
                  visibleServers[0].health.category,
                  visibleServers[0].health.message,
                ),
              }}
              components={{
                settingsLink: (
                  <Link
                    className="underline font-semibold"
                    to="/settings/mcp"
                  />
                ),
              }}
            />
          </p>
        ) : (
          <>
            <p className="text-sm font-medium">
              {t(I18nKey.CONVERSATION$MCP_UNAVAILABLE_MULTIPLE, {
                count: visibleServers.length,
              })}
            </p>
            <ul className="text-sm list-disc pl-5 space-y-0.5">
              {visibleServers.map(({ serverId, health }) => (
                <li key={serverId}>
                  {t(I18nKey.CONVERSATION$MCP_UNAVAILABLE_SERVER_LINE, {
                    serverName: serverId,
                    detail: getMcpFailureLabel(
                      t,
                      health.category,
                      health.message,
                    ),
                  })}
                </li>
              ))}
            </ul>
            <p className="text-sm">
              <Trans
                i18nKey={I18nKey.CONVERSATION$MCP_UNAVAILABLE_SETTINGS_LINK}
                components={{
                  settingsLink: (
                    <Link
                      className="underline font-semibold"
                      to="/settings/mcp"
                    />
                  ),
                }}
              />
            </p>
          </>
        )}
      </div>
      <button
        type="button"
        onClick={dismissAll}
        className="shrink-0 rounded-md p-1 hover:bg-black/10 cursor-pointer"
        aria-label={t(I18nKey.BUTTON$CLOSE)}
        data-testid="mcp-unavailable-banner-dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
