import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import { seedMcpServerHealth } from "#/api/mcp-health/probe-mcp-server-health";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { useMcpServerHealth } from "#/hooks/use-mcp-server-health";
import type { McpServerHealth } from "#/types/mcp-health";
import { MCPServerConfig } from "#/types/mcp-server";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import type { MarketplaceEntry } from "#/utils/mcp-marketplace-utils";
import { makeMcpTestErrorMessage } from "#/utils/mcp-test-error-message";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { cn } from "#/utils/utils";

interface McpServerHealthSectionProps {
  server: MCPServerConfig;
  catalog: MarketplaceEntry | undefined;
  onEdit: () => void;
}

const DOT_CLASS_BY_STATUS: Record<McpServerHealth["status"], string> = {
  unchecked: "bg-[var(--oh-text-tertiary)]",
  checking: "bg-[var(--oh-interactive-selected)] animate-pulse",
  healthy: "bg-[var(--oh-status-success)]",
  failed: "bg-red-500",
};

const actionClassName =
  "text-xs text-[var(--oh-muted)] underline transition-colors hover:text-white disabled:cursor-not-allowed disabled:opacity-60";

function getDotStatus(health: McpServerHealth): string {
  if (health.status === "healthy") {
    return health.verification === "verified"
      ? "healthy"
      : "healthy-connectivity";
  }
  return health.status;
}

function getStatusLabel(
  t: TFunction<"openhands">,
  health: McpServerHealth,
): string {
  switch (health.status) {
    case "checking":
      return t(I18nKey.MCP$HEALTH_STATUS_CHECKING);
    case "healthy":
      return health.verification === "verified"
        ? t(I18nKey.MCP$HEALTH_STATUS_VERIFIED, { count: health.toolCount })
        : t(I18nKey.MCP$HEALTH_STATUS_CONNECTIVITY_ONLY);
    case "failed":
      return makeMcpTestErrorMessage(t, health.kind, health.error);
    default:
      return t(I18nKey.MCP$HEALTH_STATUS_UNCHECKED);
  }
}

/**
 * Connection-health row for an installed server card: status dot + label,
 * a Test connection / Retry action, and — on failure — the relevant
 * recovery actions (fix credentials, re-run OAuth, open the docs).
 *
 * Renders nothing for cloud backends: the test endpoint only exists on the
 * local agent-server (`McpService.testServer` short-circuits cloud with a
 * synthetic success that must not be presented as a health verdict).
 */
export function McpServerHealthSection({
  server,
  catalog,
  onEdit,
}: McpServerHealthSectionProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { health, probe, reauthorize } = useMcpServerHealth(server);
  const { mutate: updateMcpServer } = useUpdateMcpServer();

  if (backend.kind === "cloud") return null;

  const isChecking = health.status === "checking";
  const isFailed = health.status === "failed";
  const label = getStatusLabel(t, health);

  const handleReauthorize = async () => {
    const result = await reauthorize();
    if (
      !result?.ok ||
      !result.oauth_state ||
      server.auth?.strategy !== "oauth2"
    ) {
      return;
    }
    // Persist the refreshed tokens, mirroring the editor's OAuth save path.
    const serverToSave: MCPServerConfig = {
      ...server,
      auth: { ...server.auth, state: result.oauth_state },
    };
    updateMcpServer(
      { serverId: server.id, server: serverToSave },
      {
        // The update mutation clears this server's health entry; re-seed it
        // from the probe that just succeeded so the verdict survives the save.
        onSuccess: () => seedMcpServerHealth(serverToSave, result, []),
        onError: (err: unknown) =>
          displayErrorToast(
            retrieveAxiosErrorMessage(err as AxiosError) ||
              t(I18nKey.ERROR$GENERIC),
          ),
      },
    );
  };

  return (
    // Propagation stopper: the card root is a role="button" that opens the
    // editor on click/Enter; interactions inside the health row must not
    // double as "edit server".
    <div
      data-testid={`mcp-server-health-${server.id}`}
      role="presentation"
      className="flex flex-col gap-1.5"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <div className="flex items-center gap-2">
        <span
          data-testid="mcp-health-dot"
          data-status={getDotStatus(health)}
          aria-hidden="true"
          className={cn(
            "inline-block h-2 w-2 shrink-0 rounded-full",
            DOT_CLASS_BY_STATUS[health.status],
          )}
        />
        <p
          data-testid={`mcp-health-label-${server.id}`}
          className={cn(
            "line-clamp-2 break-words text-xs",
            isFailed ? "text-red-500" : "text-tertiary-alt",
          )}
          title={label}
        >
          {label}
        </p>
      </div>

      {health.status === "healthy" &&
      health.verification === "connectivity-only" ? (
        <p className="text-xs text-tertiary-light">
          {t(I18nKey.MCP$HEALTH_CONNECTIVITY_ONLY_HINT)}
        </p>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          data-testid={`mcp-health-probe-${server.id}`}
          className={actionClassName}
          onClick={() => void probe()}
          disabled={isChecking}
        >
          {isFailed ? t(I18nKey.MCP$HEALTH_RETRY) : t(I18nKey.MCP$TEST_BUTTON)}
        </button>
        {isFailed && health.kind === "credentials" ? (
          <button
            type="button"
            data-testid={`mcp-health-update-credentials-${server.id}`}
            className={actionClassName}
            onClick={onEdit}
          >
            {t(I18nKey.MCP$HEALTH_UPDATE_CREDENTIALS)}
          </button>
        ) : null}
        {isFailed && server.auth?.strategy === "oauth2" ? (
          <button
            type="button"
            data-testid={`mcp-health-reauthorize-${server.id}`}
            className={actionClassName}
            onClick={() => void handleReauthorize()}
          >
            {t(I18nKey.MCP$HEALTH_REAUTHORIZE)}
          </button>
        ) : null}
        {isFailed && catalog?.docsUrl ? (
          <a
            href={catalog.docsUrl}
            target="_blank"
            rel="noreferrer"
            className={actionClassName}
          >
            {t(I18nKey.MCP$VIEW_DOCS)}
          </a>
        ) : null}
      </div>
    </div>
  );
}
