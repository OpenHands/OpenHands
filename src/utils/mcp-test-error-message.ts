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
  // Failed provider requests can append an entire HTML response body. Keep
  // the useful failure detail without displaying the document in the UI.
  const htmlStart = error.search(/<(?:!doctype\s+html|html)(?=[\s>])/i);
  const detail = htmlStart === -1 ? error : error.slice(0, htmlStart).trimEnd();

  switch (errorKind) {
    case "timeout":
      return t(I18nKey.MCP$TEST_ERROR_TIMEOUT);
    case "authorization":
      return t(I18nKey.MCP$TEST_ERROR_AUTHORIZATION);
    case "connection":
      return serverType === "stdio"
        ? t(I18nKey.MCP$TEST_ERROR_STDIO_START, { error: detail })
        : t(I18nKey.MCP$TEST_ERROR_CONNECTION, { error: detail });
    case "credentials":
      return t(I18nKey.MCP$TEST_ERROR_CREDENTIALS, { error: detail });
    default:
      return t(I18nKey.MCP$TEST_ERROR_UNKNOWN, { error: detail });
  }
}
