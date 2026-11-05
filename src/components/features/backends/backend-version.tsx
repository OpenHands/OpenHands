import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";

import { ServerClient } from "@openhands/typescript-client/clients";
import { type Backend } from "#/api/backend-registry/types";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getDisplayAgentServerVersion } from "#/api/agent-server-compatibility";
import { I18nKey } from "#/i18n/declaration";

type ExecutionMode = "local" | "docker";

function executionModeLabel(
  t: (key: I18nKey) => string,
  mode: ExecutionMode,
): string {
  return mode === "docker"
    ? t(I18nKey.BACKEND$EXECUTION_MODE_DOCKER)
    : t(I18nKey.BACKEND$EXECUTION_MODE_LOCAL);
}

// @spec BM-004 — Show the execution boundary the backend reports, so an
// operator can tell whether conversations run in-process or in Docker.
export function BackendVersion({ backend }: { backend: Backend }) {
  const { t } = useTranslation("openhands");
  const { data } = useQuery({
    queryKey: ["backend-version", backend.host, backend.apiKey],
    queryFn: async () => {
      const info = await new ServerClient(
        getAgentServerClientOptions({
          host: backend.host,
          sessionApiKey: backend.apiKey || null,
          timeout: 5000,
        }),
      ).getServerInfo();
      return {
        version: getDisplayAgentServerVersion(info),
        executionMode: info.execution_runtime as ExecutionMode | undefined,
      };
    },
    retry: false,
    staleTime: 60_000,
    enabled: backend.kind === "local",
    meta: { disableToast: true },
  });

  const version = data?.version;
  if (!version) return null;

  const executionMode = data?.executionMode;
  const modeKnown = executionMode === "local" || executionMode === "docker";

  return (
    <span
      className="inline-flex shrink-0 items-center gap-1 rounded-full border border-border bg-surface px-1.5 py-0.5 text-3xs font-medium leading-none text-text-dim"
      data-testid={`manage-backends-version-${backend.name}`}
    >
      <span>{t(I18nKey.BACKEND$VERSION_LABEL, { version })}</span>
      {modeKnown && (
        <span
          className="border-l border-border pl-1"
          data-testid={`manage-backends-execution-mode-${backend.name}`}
        >
          {executionModeLabel(t, executionMode)}
        </span>
      )}
    </span>
  );
}
