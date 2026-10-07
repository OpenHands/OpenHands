import { useTranslation } from "react-i18next";

import { type Backend } from "#/api/backend-registry/types";
import {
  type BackendExecutionMode as BackendExecutionModeValue,
  getBackendExecutionMode,
} from "#/api/agent-server-compatibility";
import { useBackendServerInfo } from "#/hooks/query/use-backend-server-info";
import { I18nKey } from "#/i18n/declaration";

const MODE_LABEL_KEYS: Record<BackendExecutionModeValue, I18nKey> = {
  local: I18nKey.BACKEND$EXECUTION_MODE_LOCAL,
  docker: I18nKey.BACKEND$EXECUTION_MODE_DOCKER,
};

/**
 * Badge showing the execution boundary (local or Docker) the backend's
 * agent-server runs conversations in. Renders nothing when the server does not
 * report a mode, so older servers are visually unchanged.
 *
 * @spec BM-004 — Display the active backend's execution mode
 */
export function BackendExecutionMode({ backend }: { backend: Backend }) {
  const { t } = useTranslation("openhands");
  const { data: serverInfo } = useBackendServerInfo(backend);
  const mode = getBackendExecutionMode(serverInfo);

  if (!mode) return null;

  return (
    <span
      className="inline-flex shrink-0 items-center rounded-full border border-border bg-surface px-1.5 py-0.5 text-[10px] font-medium leading-none text-text-dim"
      data-testid={`manage-backends-execution-mode-${backend.name}`}
      data-execution-mode={mode}
    >
      {t(MODE_LABEL_KEYS[mode])}
    </span>
  );
}
