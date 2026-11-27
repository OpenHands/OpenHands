import type { TFunction } from "i18next";
import { isAgentServerVersionError } from "@openhands/typescript-client/clients";

import {
  WORKSPACE_RENAME_MINIMUM_AGENT_SERVER_VERSION,
  compareAgentServerVersions,
  getCachedAgentServerVersion,
} from "#/api/agent-server-compatibility";
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

/**
 * Tooltip copy for a backend that predates workspace rename, or `null` when
 * the action should be offered.
 *
 * Returns `null` for an unknown or unparseable version on purpose: the pencil
 * stays usable and the typed version error surfaces on click as a backstop.
 * Only a *known-old* backend disables it.
 */
export function getWorkspaceRenameUnsupportedMessage(
  t: TFunction<"openhands">,
): string | null {
  const actualVersion = getCachedAgentServerVersion();
  if (!actualVersion) {
    return null;
  }

  const comparison = compareAgentServerVersions(
    actualVersion,
    WORKSPACE_RENAME_MINIMUM_AGENT_SERVER_VERSION,
  );
  if (comparison === null || comparison >= 0) {
    return null;
  }

  return t(I18nKey.HOME$WORKSPACES_UNSUPPORTED_AGENT_SERVER, {
    actualVersion,
    requiredVersion: WORKSPACE_RENAME_MINIMUM_AGENT_SERVER_VERSION,
  });
}
