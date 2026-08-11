import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";

import { type Backend } from "#/api/backend-registry/types";
import { fetchAgentServerVersion } from "#/api/backend-registry/backend-connection-service";
import { I18nKey } from "#/i18n/declaration";

export function BackendVersion({ backend }: { backend: Backend }) {
  const { t } = useTranslation("openhands");
  const { data: version } = useQuery({
    queryKey: ["backend-version", backend.host, backend.apiKey],
    queryFn: () =>
      fetchAgentServerVersion({ host: backend.host, apiKey: backend.apiKey }),
    retry: false,
    staleTime: 60_000,
    enabled: backend.kind === "local",
  });

  if (!version) return null;

  return (
    <span
      className="inline-flex shrink-0 items-center rounded-full border border-[var(--oh-border)] bg-[var(--oh-surface)] px-1.5 py-0.5 text-[10px] font-medium leading-none text-[var(--oh-text-dim)]"
      data-testid={`manage-backends-version-${backend.name}`}
    >
      {t(I18nKey.BACKEND$VERSION_LABEL, { version })}
    </span>
  );
}
