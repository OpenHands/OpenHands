import { useTranslation } from "react-i18next";

import { type Backend } from "#/api/backend-registry/types";
import { useBackendVersion } from "#/hooks/query/use-backend-server-info";
import { I18nKey } from "#/i18n/declaration";

export function BackendVersion({ backend }: { backend: Backend }) {
  const { t } = useTranslation("openhands");
  const version = useBackendVersion(backend);

  if (!version) return null;

  return (
    <span
      className="inline-flex shrink-0 items-center rounded-full border border-border bg-surface px-1.5 py-0.5 text-[10px] font-medium leading-none text-text-dim"
      data-testid={`manage-backends-version-${backend.name}`}
    >
      {t(I18nKey.BACKEND$VERSION_LABEL, { version })}
    </span>
  );
}
