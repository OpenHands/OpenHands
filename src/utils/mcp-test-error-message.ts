import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import type { ExtendedMCPTestFailureKind } from "#/types/mcp-server";

/**
 * Kind-specific, localized guidance for a failed MCP connection test.
 * `error` is interpolated for the kinds whose message surfaces the provider
 * detail — callers pass display-safe (redacted) text.
 */
export function makeMcpTestErrorMessage(
  t: TFunction<"openhands">,
  errorKind: ExtendedMCPTestFailureKind,
  error: string,
): string {
  switch (errorKind) {
    case "timeout":
      return t(I18nKey.MCP$TEST_ERROR_TIMEOUT);
    case "connection":
      return t(I18nKey.MCP$TEST_ERROR_CONNECTION);
    case "credentials":
      return t(I18nKey.MCP$TEST_ERROR_CREDENTIALS, { error });
    default:
      return t(I18nKey.MCP$TEST_ERROR_UNKNOWN, { error });
  }
}
