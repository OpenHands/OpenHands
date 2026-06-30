import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import {
  McpServerFailureCategory,
  McpServerHealthStatus,
} from "#/types/mcp-test";
import { Typography } from "#/ui/typography";

const FAILURE_I18N: Record<McpServerFailureCategory, I18nKey> = {
  configuration: I18nKey.SETTINGS$MCP_FAILURE_CONFIGURATION,
  connection: I18nKey.SETTINGS$MCP_FAILURE_CONNECTION,
  authentication: I18nKey.SETTINGS$MCP_FAILURE_AUTHENTICATION,
  protocol: I18nKey.SETTINGS$MCP_FAILURE_PROTOCOL,
  tool_discovery: I18nKey.SETTINGS$MCP_FAILURE_TOOL_DISCOVERY,
  execution: I18nKey.SETTINGS$MCP_FAILURE_EXECUTION,
  sandbox: I18nKey.SETTINGS$MCP_FAILURE_SANDBOX,
  timeout: I18nKey.SETTINGS$MCP_FAILURE_TIMEOUT,
  internal: I18nKey.SETTINGS$MCP_FAILURE_INTERNAL,
};

export interface McpServerHealthBadgeProps {
  status: McpServerHealthStatus;
  category?: McpServerFailureCategory | null;
  message?: string | null;
}

export function McpServerHealthBadge({
  status,
  category,
  message,
}: McpServerHealthBadgeProps) {
  const { t } = useTranslation();

  if (status === "testing") {
    return (
      <div className="flex items-center gap-2">
        <LoadingSpinner size="small" />
        <Typography.Text className="text-xs text-content-2">
          {t(I18nKey.SETTINGS$MCP_HEALTH_TESTING)}
        </Typography.Text>
      </div>
    );
  }

  if (status === "healthy") {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
        {t(I18nKey.SETTINGS$MCP_HEALTH_HEALTHY)}
      </Typography.Text>
    );
  }

  if (status === "unhealthy") {
    const label = category
      ? t(FAILURE_I18N[category])
      : t(I18nKey.SETTINGS$MCP_HEALTH_UNHEALTHY);
    return (
      <span title={message || undefined}>
        <Typography.Text className="px-2 py-1 text-xs rounded bg-red-500/20 text-red-400">
          {label}
        </Typography.Text>
      </span>
    );
  }

  return (
    <Typography.Text className="px-2 py-1 text-xs rounded bg-gray-500/20 text-gray-400">
      {t(I18nKey.SETTINGS$MCP_HEALTH_UNKNOWN)}
    </Typography.Text>
  );
}
