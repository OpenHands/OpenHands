import React from "react";
import { Puzzle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import { MCPServerConfig } from "#/types/mcp-server";
import { INTEGRATION_CATALOG as MCP_MARKETPLACE } from "@openhands/extensions/integrations";
import {
  findCatalogEntryForServer,
  getMcpMarketplaceCatalog,
} from "#/utils/mcp-marketplace-utils";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { getInstalledServerTitle } from "#/utils/mcp-installed-server-display";
import { McpServerHealthSection } from "./mcp-server-health";

interface InstalledServerCardProps {
  server: MCPServerConfig;
  onEdit: () => void;
  onDelete: () => void;
}

function getServerTransportLabel(type: MCPServerConfig["type"]) {
  switch (type) {
    case "sse":
      return "SSE";
    case "shttp":
      return "HTTP";
    case "stdio":
      return "STDIO";
    default:
      return type;
  }
}

function getServerDetailLine(server: MCPServerConfig): string {
  if (server.type === "stdio") {
    const args =
      server.args && server.args.length > 0 ? ` ${server.args.join(" ")}` : "";
    return `${server.command ?? ""}${args}`.trim();
  }
  return server.url ?? "";
}

export function InstalledServerCard({
  server,
  onEdit,
  onDelete,
}: InstalledServerCardProps) {
  const { t } = useTranslation("openhands");
  const catalog = findCatalogEntryForServer(
    server,
    getMcpMarketplaceCatalog(MCP_MARKETPLACE),
  );

  const title = getInstalledServerTitle(server, catalog);
  const detailLine = getServerDetailLine(server);
  const transport = getServerTransportLabel(server.type);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onEdit();
    }
  };

  return (
    <div
      data-testid="mcp-server-item"
      data-server-id={server.id}
      role="button"
      tabIndex={0}
      onClick={onEdit}
      onKeyDown={handleKeyDown}
      aria-label={t(I18nKey.MCP$EDIT_SERVER_ARIA, { name: title })}
      className={cn(
        "flex min-h-[132px] flex-col overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <div className="flex items-start gap-3">
        <McpLogoBadge
          entry={catalog}
          fallback={<Puzzle strokeWidth={2.25} />}
        />

        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <header className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-sm font-semibold" title={title}>
                {title}
              </h3>
              <p className="mt-0.5 text-xs text-tertiary-alt">{transport}</p>
            </div>
            <CirclePlusCheckToggle
              testId={`mcp-installed-toggle-${server.id}`}
              isSelected
              onToggle={(selected) => {
                if (!selected) {
                  onDelete();
                }
              }}
              enableLabelKey={I18nKey.MCP$TOGGLE_ADD_SERVER}
              disableLabelKey={I18nKey.MCP$TOGGLE_REMOVE_SERVER}
            />
          </header>

          {catalog?.description ? (
            <p
              data-testid={`mcp-server-description-${server.id}`}
              className="line-clamp-2 break-words text-xs leading-relaxed text-tertiary-light"
            >
              {catalog.description}
            </p>
          ) : null}

          {detailLine ? (
            <p
              data-testid={`mcp-server-detail-${server.id}`}
              className="truncate text-xs text-tertiary-alt"
              title={detailLine}
            >
              {detailLine}
            </p>
          ) : null}

          <McpServerHealthSection
            server={server}
            catalog={catalog}
            onEdit={onEdit}
          />
        </div>
      </div>
    </div>
  );
}
