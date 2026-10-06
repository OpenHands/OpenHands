import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import type {
  ExtendedMCPTestFailureKind,
  MCPServerType,
} from "#/types/mcp-server";

/**
 * Kind-specific, localized guidance for a failed MCP connection test.
 * `error` is interpolated for the kinds whose message surfaces the provider
 * detail — callers pass display-safe (redacted) text. `serverType` picks
 * transport-appropriate copy for a `connection` failure: a stdio server has
 * no URL, its command failed to start. An unknown type reads as remote.
 */
export function makeMcpTestErrorMessage(
  t: TFunction<"openhands">,
  errorKind: ExtendedMCPTestFailureKind,
  error: string,
  serverType: MCPServerType | undefined,
): string {
  switch (errorKind) {
    case "timeout":
      return t(I18nKey.MCP$TEST_ERROR_TIMEOUT);
    case "connection":
      return serverType === "stdio"
        ? t(I18nKey.MCP$TEST_ERROR_STDIO_START, { error })
        : t(I18nKey.MCP$TEST_ERROR_CONNECTION, { error });
    case "credentials":
      return t(I18nKey.MCP$TEST_ERROR_CREDENTIALS, { error });
    default:
      return t(I18nKey.MCP$TEST_ERROR_UNKNOWN, { error });
  }
}
