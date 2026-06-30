import { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import { McpServerFailureCategory } from "#/types/mcp-test";

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

export function getMcpFailureLabel(
  t: TFunction,
  category?: McpServerFailureCategory | null,
  message?: string | null,
): string {
  if (message?.trim()) {
    return message.trim();
  }
  if (category) {
    return String(t(FAILURE_I18N[category]));
  }
  return String(t(I18nKey.SETTINGS$MCP_HEALTH_UNHEALTHY));
}
