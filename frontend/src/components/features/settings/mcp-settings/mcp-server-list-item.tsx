import { FaPencil, FaTrash, FaPlug } from "react-icons/fa6";
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useTestMcpServer } from "#/hooks/mutation/use-test-mcp-server";
import { useMcpServerHealth } from "#/hooks/query/use-mcp-server-health";
import { useMcpTestRun } from "#/hooks/query/use-mcp-test-run";
import { McpServerHealthStatus } from "#/types/mcp-test";
import { getMcpServerId, McpServerListEntry } from "#/utils/mcp-server-id";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { McpServerHealthBadge } from "./mcp-server-health-badge";

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

export function MCPServerListItem({
  server,
  pendingTestId = null,
  onPendingTestComplete,
  onEdit,
  onDelete,
}: {
  server: McpServerListEntry;
  pendingTestId?: string | null;
  onPendingTestComplete?: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const serverId = getMcpServerId(server);
  const [activeTestId, setActiveTestId] = useState<string | null>(
    pendingTestId,
  );

  const { data: health } = useMcpServerHealth(serverId);
  const { data: testRun } = useMcpTestRun(activeTestId);
  const { mutate: startTest, isPending: isStartingTest } = useTestMcpServer();

  useEffect(() => {
    if (pendingTestId) {
      setActiveTestId(pendingTestId);
    }
  }, [pendingTestId]);

  useEffect(() => {
    if (testRun && TERMINAL.has(testRun.status)) {
      queryClient.invalidateQueries({
        queryKey: ["mcp-server-health", serverId],
      });
      setActiveTestId(null);
      onPendingTestComplete?.();
    }
  }, [testRun, serverId, queryClient, onPendingTestComplete]);

  const getServerTypeLabel = (type: string) => {
    switch (type) {
      case "sse":
        return t(I18nKey.SETTINGS$MCP_SERVER_TYPE_SSE);
      case "stdio":
        return t(I18nKey.SETTINGS$MCP_SERVER_TYPE_STDIO);
      case "shttp":
        return t(I18nKey.SETTINGS$MCP_SERVER_TYPE_SHTTP);
      default:
        return type.toUpperCase();
    }
  };

  const getServerDescription = (serverConfig: McpServerListEntry) => {
    if (serverConfig.type === "stdio") {
      if (serverConfig.command) {
        const args =
          serverConfig.args && serverConfig.args.length > 0
            ? ` ${serverConfig.args.join(" ")}`
            : "";
        return `${serverConfig.command}${args}`;
      }
      return serverConfig.name || "";
    }
    if (
      (serverConfig.type === "sse" || serverConfig.type === "shttp") &&
      serverConfig.url
    ) {
      return serverConfig.url;
    }
    return "";
  };

  const displayName =
    server.type === "stdio"
      ? server.name
      : server.name || server.url || server.id;
  const serverDescription = getServerDescription(server);

  const isTesting =
    isStartingTest ||
    (!!activeTestId && testRun?.status === "running") ||
    health?.status === "testing";

  let badgeStatus: McpServerHealthStatus = health?.status ?? "unknown";
  if (isTesting) {
    badgeStatus = "testing";
  } else if (testRun && TERMINAL.has(testRun.status)) {
    badgeStatus = testRun.status === "succeeded" ? "healthy" : "unhealthy";
  }

  const badgeCategory = testRun?.category ?? health?.category ?? null;
  const badgeMessage = testRun?.message ?? health?.message ?? null;

  const lastTestedAt =
    (testRun && TERMINAL.has(testRun.status)
      ? (testRun.finished_at ?? testRun.started_at ?? testRun.created_at)
      : undefined) ??
    health?.tested_at ??
    undefined;
  const lastTestedLabel = lastTestedAt
    ? `${formatTimeDelta(lastTestedAt)} ${t(I18nKey.CONVERSATION$AGO)}`
    : null;

  const handleTest = () => {
    startTest(serverId, {
      onSuccess: (response) => {
        setActiveTestId(response.test_id);
      },
    });
  };

  return (
    <tr
      data-testid="mcp-server-item"
      className="grid grid-cols-[minmax(0,0.2fr)_100px_minmax(0,1fr)_120px_120px_160px] gap-4 items-start border-t border-tertiary"
    >
      <td
        className="p-3 text-sm text-content-2 truncate min-w-0"
        title={displayName}
      >
        {displayName}
      </td>

      <td className="p-3 text-sm text-content-2 whitespace-nowrap">
        {getServerTypeLabel(server.type)}
      </td>

      <td
        className="p-3 text-sm text-content-2 opacity-80 italic min-w-0 truncate"
        title={serverDescription}
      >
        <span className="inline-block max-w-full align-bottom">
          {serverDescription}
        </span>
      </td>

      <td className="p-3">
        <McpServerHealthBadge
          status={badgeStatus}
          category={badgeCategory}
          message={badgeMessage}
        />
      </td>

      <td
        className="p-3 text-sm text-content-2 whitespace-nowrap"
        data-testid="mcp-server-last-tested"
        title={lastTestedAt}
      >
        {lastTestedLabel ? (
          <time dateTime={lastTestedAt}>{lastTestedLabel}</time>
        ) : (
          "—"
        )}
      </td>

      <td className="p-3 flex items-start justify-end gap-3 whitespace-nowrap">
        <button
          data-testid="test-mcp-server-button"
          type="button"
          onClick={handleTest}
          disabled={isTesting}
          aria-label={`Test ${displayName}`}
          title={t(I18nKey.SETTINGS$MCP_TEST_CONNECTION)}
          className="cursor-pointer hover:text-content-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <FaPlug size={16} />
        </button>
        <button
          data-testid="edit-mcp-server-button"
          type="button"
          onClick={onEdit}
          aria-label={`Edit ${displayName}`}
          className="cursor-pointer hover:text-content-1 transition-colors"
        >
          <FaPencil size={16} />
        </button>
        <button
          data-testid="delete-mcp-server-button"
          type="button"
          onClick={onDelete}
          aria-label={`Delete ${displayName}`}
          className="cursor-pointer hover:text-content-1 transition-colors"
        >
          <FaTrash size={16} />
        </button>
      </td>
    </tr>
  );
}
