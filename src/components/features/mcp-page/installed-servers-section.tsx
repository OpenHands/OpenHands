import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MCPServerConfig } from "#/types/mcp-server";
import { InstalledServerCard } from "./installed-server-card";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";

interface InstalledServersSectionProps {
  /** Already-filtered list — search filtering happens upstream. */
  servers: MCPServerConfig[];
  /**
   * True iff there is at least one installed server before applying
   * the search filter. Lets the section differentiate "nothing
   * installed yet" from "no installed servers match the current
   * search".
   */
  hasAnyInstalled: boolean;
  /** Current search query — empty string means no filter applied. */
  query?: string;
  onEdit: (server: MCPServerConfig) => void;
  onDelete: (serverId: string) => void;
}

export function InstalledServersSection({
  servers,
  hasAnyInstalled,
  query = "",
  onEdit,
  onDelete,
}: InstalledServersSectionProps) {
  const { t } = useTranslation("openhands");

  const isEmpty = servers.length === 0;

  if (isEmpty) {
    // Filter narrowed everything out — vs. nothing was installed in
    // the first place. Different copy in each case.
    if (hasAnyInstalled && query.trim().length > 0) {
      return (
        <div
          data-testid="mcp-installed-empty-search"
          className="rounded-xl border border-[var(--oh-border)] p-6 text-center"
        >
          <p className="text-xs text-tertiary-light">
            {t(I18nKey.MCP$SEARCH_EMPTY)}
          </p>
        </div>
      );
    }
    return (
      <div
        data-testid="mcp-installed-empty"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-white">
          {t(I18nKey.MCP$INSTALLED_EMPTY_TITLE)}
        </p>
        <p className="text-xs text-tertiary-light mt-1">
          {t(I18nKey.MCP$INSTALLED_EMPTY_HINT)}
        </p>
      </div>
    );
  }

  return (
    <div className={extensionModuleCardGridContainerClassName}>
      <div
        data-testid="mcp-installed-list"
        className={extensionModuleCardGridClassName}
      >
        {servers.map((server) => (
          <InstalledServerCard
            key={server.id}
            server={server}
            onEdit={() => onEdit(server)}
            onDelete={() => onDelete(server.id)}
          />
        ))}
      </div>
    </div>
  );
}
