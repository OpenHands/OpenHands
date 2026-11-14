import React from "react";
import { ChevronDown, Puzzle } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  INTEGRATION_CATALOG,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { dropdownMenuListClassName } from "#/utils/dropdown-classes";
import { getInstalledServerTitle } from "#/utils/mcp-installed-server-display";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import {
  findCatalogEntryForServer,
  getMcpMarketplaceCatalog,
} from "#/utils/mcp-marketplace-utils";
import { cn } from "#/utils/utils";

interface McpServerCheckboxSelectorProps {
  selectedRefs: Set<string>;
  onSelectedRefsChange: (refs: Set<string>) => void;
  disabled?: boolean;
  testId?: string;
}

export function McpServerCheckboxSelector({
  selectedRefs,
  onSelectedRefsChange,
  disabled = false,
  testId = "mcp-server-checkbox-selector",
}: McpServerCheckboxSelectorProps) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading, isError, refetch } = useSettings();
  const [open, setOpen] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const catalog = React.useMemo(
    () => getMcpMarketplaceCatalog(INTEGRATION_CATALOG),
    [],
  );
  const servers = flattenMcpConfig(settings?.mcp_config ?? {}).filter(
    (server) => server.enabled !== false,
  );
  const unavailable = disabled || isLoading || isError || servers.length === 0;

  React.useEffect(() => {
    if (!open) return undefined;
    const onMouseDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [open]);

  React.useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const toggle = (ref: string, checked: boolean) => {
    const next = new Set(selectedRefs);
    if (checked) next.add(ref);
    else next.delete(ref);
    onSelectedRefsChange(next);
  };

  const titleFor = (
    server: (typeof servers)[number],
    entry: MarketplaceEntry | undefined,
  ) => getInstalledServerTitle(server, entry);

  return (
    <div ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        data-testid={testId}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t(I18nKey.MCP$INSTALLED_TITLE)}
        disabled={disabled || isLoading}
        onClick={() => setOpen((next) => !next)}
        className={cn(
          "group flex flex-row items-center justify-between gap-2 rounded-pill border border-border-input bg-transparent py-1 pl-2.5 pr-2 text-contrast dark:border-border-subtle",
          disabled || isLoading
            ? "cursor-not-allowed opacity-50"
            : "cursor-pointer hover:border-muted dark:hover:border-border-subtle",
        )}
      >
        <Puzzle className="size-3 shrink-0" aria-hidden />
        <span className="min-w-0 flex-1 truncate text-sm font-normal leading-5">
          {t(I18nKey.MCP$INSTALLED_TITLE)}
          {selectedRefs.size > 0 ? ` (${selectedRefs.size})` : ""}
        </span>
        <ChevronDown className="size-3 shrink-0 text-contrast" aria-hidden />
      </button>

      {open ? (
        <div
          role="menu"
          data-testid={`${testId}-menu`}
          className={cn(
            "absolute left-0 top-full z-50 mt-1 w-72 rounded-lg border border-border bg-surface p-2 shadow-lg",
            dropdownMenuListClassName,
          )}
        >
          {isError ? (
            <button
              type="button"
              className="text-left text-sm text-primary underline"
              onClick={() => void refetch()}
            >
              {t(I18nKey.MCP$HEALTH_RETRY)}
            </button>
          ) : servers.length === 0 ? (
            <p className="px-2 py-1 text-sm text-muted">
              {t(I18nKey.MCP$INSTALLED_EMPTY_TITLE)}
            </p>
          ) : (
            <ul className="max-h-60 space-y-1 overflow-y-auto">
              {servers.map((server) => {
                const entry = findCatalogEntryForServer(server, catalog);
                const title = titleFor(server, entry);
                return (
                  <li key={server.id}>
                    <label className="flex cursor-pointer items-center gap-3 rounded-md px-2 py-1.5 hover:bg-tertiary">
                      <input
                        type="checkbox"
                        className="size-4 shrink-0 accent-primary"
                        checked={selectedRefs.has(server.id)}
                        disabled={unavailable}
                        onChange={(event) =>
                          toggle(server.id, event.target.checked)
                        }
                        aria-label={t(I18nKey.MCP$ATTACH_TO_CONVERSATION, {
                          name: title,
                        })}
                      />
                      <McpLogoBadge
                        entry={entry}
                        size="base"
                        fallback={<Puzzle size={16} />}
                      />
                      <span
                        className="min-w-0 flex-1 truncate text-sm"
                        title={title}
                      >
                        {title}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ) : null}
    </div>
  );
}
