import { useTranslation } from "react-i18next";
import { MCPServerListItem } from "./mcp-server-list-item";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import type { MCPServerConfig } from "#/types/mcp-server";

interface MCPServerListProps {
  servers: MCPServerConfig[];
  onEdit: (server: MCPServerConfig) => void;
  onDelete: (serverId: string) => void;
}

export function MCPServerList({
  servers,
  onEdit,
  onDelete,
}: MCPServerListProps) {
  const { t } = useTranslation("openhands");

  if (servers.length === 0) {
    return (
      <div className={extensionModuleEmptyStateClassName}>
        <p className="text-content-2 text-sm">
          {t(I18nKey.SETTINGS$MCP_NO_SERVERS)}
        </p>
      </div>
    );
  }

  return (
    <div className="border border-[var(--oh-border)] rounded-md overflow-hidden">
      <table className="w-full">
        <thead className="bg-base-tertiary">
          <tr className="grid grid-cols-[minmax(0,0.25fr)_120px_minmax(0,1fr)_120px] gap-4 items-start">
            <th className="px-3 py-2 text-left text-sm font-medium">
              {t(I18nKey.SETTINGS$NAME)}
            </th>
            <th className="px-3 py-2 text-left text-sm font-medium">
              {t(I18nKey.SETTINGS$MCP_SERVER_TYPE)}
            </th>
            <th className="px-3 py-2 text-left text-sm font-medium">
              {t(I18nKey.SETTINGS$MCP_SERVER_DETAILS)}
            </th>
            <th className="px-3 py-2 text-right text-sm font-medium">
              {t(I18nKey.SETTINGS$ACTIONS)}
            </th>
          </tr>
        </thead>
        <tbody>
          {servers.map((server) => (
            <MCPServerListItem
              key={server.id}
              server={server}
              onEdit={() => onEdit(server)}
              onDelete={() => onDelete(server.id)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}
