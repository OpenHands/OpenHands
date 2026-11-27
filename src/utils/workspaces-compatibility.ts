import type { TFunction } from "i18next";
import { isAgentServerVersionError } from "@openhands/typescript-client/clients";

import { I18nKey } from "#/i18n/declaration";

export function getWorkspacesUnsupportedMessage(
  error: unknown,
  t: TFunction<"openhands">,
) {
  if (!isAgentServerVersionError(error)) {
    return null;
  }

  return t(I18nKey.HOME$WORKSPACES_UNSUPPORTED_AGENT_SERVER, {
    actualVersion: error.actualVersion,
    requiredVersion: error.requiredVersion,
  });
}
