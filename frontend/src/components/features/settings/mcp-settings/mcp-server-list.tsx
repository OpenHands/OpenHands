import { useTranslation } from "react-i18next";
import { MCPServerListItem } from "./mcp-server-list-item";
import { I18nKey } from "#/i18n/declaration";
import { getMcpServerId } from "#/utils/mcp-server-id";

interface MCPServerConfig {
  id: string;
  type: "sse" | "stdio" | "shttp";
  name?: string;
  url?: string;
  api_key?: string;
  timeout?: number;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
}

interface MCPServerListProps {
  servers: MCPServerConfig[];
  pendingTestsByServerId?: Record<string, string>;
  onPendingTestComplete?: (serverId: string) => void;
  onEdit: (server: MCPServerConfig) => void;
  onDelete: (serverId: string) => void;
}

export function MCPServerList({
  servers,
  pendingTestsByServerId,
  onPendingTestComplete,
  onEdit,
  onDelete,
}: MCPServerListProps) {
  const { t } = useTranslation();

  if (servers.length === 0) {
    return (
      <div className="border border-tertiary rounded-md p-8 text-center">
        <p className="text-content-2 text-sm">
          {t(I18nKey.SETTINGS$MCP_NO_SERVERS)}
        </p>
      </div>
    );
  }

  return (
    <div className="border border-tertiary rounded-md overflow-hidden">
      <table className="w-full">
        <thead className="bg-base-tertiary">
          <tr className="grid grid-cols-[minmax(0,0.2fr)_100px_minmax(0,1fr)_120px_120px_160px] gap-4 items-start">
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$NAME)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$MCP_SERVER_TYPE)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$MCP_SERVER_DETAILS)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$MCP_HEALTH_STATUS)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$MCP_LAST_TESTED)}
            </th>
            <th className="text-right p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$ACTIONS)}
            </th>
          </tr>
        </thead>
        <tbody>
          {servers.map((server) => {
            const serverId = getMcpServerId(server);
            return (
              <MCPServerListItem
                key={server.id}
                server={server}
                pendingTestId={pendingTestsByServerId?.[serverId] ?? null}
                onPendingTestComplete={
                  onPendingTestComplete
                    ? () => onPendingTestComplete(serverId)
                    : undefined
                }
                onEdit={() => onEdit(server)}
                onDelete={() => onDelete(server.id)}
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
