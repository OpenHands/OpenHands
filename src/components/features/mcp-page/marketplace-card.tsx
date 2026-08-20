import type { KeyboardEvent } from "react";
import { I18nKey } from "#/i18n/declaration";
import type { IntegrationCatalogEntry as MarketplaceEntry } from "@openhands/extensions/integrations";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import { getDefaultMcpTransport } from "#/utils/mcp-marketplace-utils";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";

interface MarketplaceCardProps {
  entry: MarketplaceEntry;
  onClick: () => void;
  onAdd: () => void;
}

export function MarketplaceCard({
  entry,
  onClick,
  onAdd,
}: MarketplaceCardProps) {
  const transport = getDefaultMcpTransport(entry);
  const transportLabel = (() => {
    switch (transport?.kind) {
      case "stdio":
        return "STDIO";
      case "shttp":
        return "HTTP";
      case "sse":
        return "SSE";
      default:
        return "";
    }
  })();

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) {
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClick();
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      data-testid={`mcp-marketplace-card-${entry.id}`}
      className={cn(
        "flex min-h-[132px] flex-col overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <div className="flex items-start gap-3">
        <McpLogoBadge entry={entry} />
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <header className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-sm font-semibold">{entry.name}</h3>
              <p className="mt-0.5 text-xs text-tertiary-alt">
                {transportLabel}
              </p>
            </div>
            <CirclePlusCheckToggle
              testId={`mcp-marketplace-toggle-${entry.id}`}
              isSelected={false}
              onToggle={(selected) => {
                if (selected) {
                  onAdd();
                }
              }}
              enableLabelKey={I18nKey.MCP$TOGGLE_ADD_SERVER}
              disableLabelKey={I18nKey.MCP$TOGGLE_ADD_SERVER}
            />
          </header>
          <p className="line-clamp-3 text-xs leading-relaxed text-tertiary-light">
            {entry.description}
          </p>
        </div>
      </div>
    </div>
  );
}
