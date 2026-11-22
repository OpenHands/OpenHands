import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import {
  AGENT_CANVAS_CLIENT_VERSION,
  OPENHANDS_EXTENSIONS_VERSION,
  OPENHANDS_TYPESCRIPT_CLIENT_VERSION,
} from "#/api/client-source";
import { useRunningVersions } from "#/hooks/query/use-running-versions";

interface VersionRowProps {
  name: string;
  version: string | null;
  notReported: string;
}

function VersionRow({ name, version, notReported }: VersionRowProps) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-border py-2 last:border-0">
      <dt className="min-w-0 text-sm text-text-tertiary">{name}</dt>
      <dd className="shrink-0 font-mono text-sm text-contrast">
        {version ?? notReported}
      </dd>
    </div>
  );
}

export function RunningVersionsSection() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const {
    backendVersions,
    automationVersions,
    isLoading,
    isFetching,
    refetch,
  } = useRunningVersions();
  const notReported = t(I18nKey.SETTINGS$RUNNING_VERSIONS_NOT_REPORTED);
  const rows: VersionRowProps[] = [
    {
      name: "@openhands/agent-canvas",
      version: AGENT_CANVAS_CLIENT_VERSION,
      notReported,
    },
    {
      name: "@openhands/typescript-client",
      version: OPENHANDS_TYPESCRIPT_CLIENT_VERSION,
      notReported,
    },
    {
      name: "@openhands/extensions",
      version: OPENHANDS_EXTENSIONS_VERSION,
      notReported,
    },
    {
      name:
        backend.kind === "cloud"
          ? t(I18nKey.SETTINGS$RUNNING_VERSIONS_CLOUD_SERVICE)
          : "openhands-agent-server",
      version: backendVersions?.serviceVersion ?? null,
      notReported,
    },
    {
      name: "openhands-sdk",
      version: backendVersions?.sdkVersion ?? null,
      notReported,
    },
    {
      name: "openhands-tools",
      version: backendVersions?.toolsVersion ?? null,
      notReported,
    },
    {
      name: "openhands-workspace",
      version: backendVersions?.workspaceVersion ?? null,
      notReported,
    },
    {
      name: "openhands-automation",
      version: automationVersions?.packageVersion ?? null,
      notReported,
    },
    {
      name: "openhands-sdk (automation)",
      version: automationVersions?.sdkVersion ?? null,
      notReported,
    },
  ];

  return (
    <section
      aria-labelledby="running-versions-heading"
      data-testid="running-versions"
      className="mt-2 border-t border-border pt-6"
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3
            id="running-versions-heading"
            className="mb-1 text-lg font-medium text-contrast"
          >
            {t(I18nKey.SETTINGS$RUNNING_VERSIONS_TITLE)}
          </h3>
          <p className="text-sm leading-5 text-tertiary-light">
            {t(I18nKey.SETTINGS$RUNNING_VERSIONS_DESCRIPTION)}
          </p>
        </div>
        <button
          type="button"
          aria-label={t(I18nKey.BUTTON$REFRESH)}
          aria-busy={isFetching}
          disabled={isFetching}
          onClick={() => void refetch()}
          className="inline-flex size-9 shrink-0 items-center justify-center rounded-md border border-border text-muted transition-colors hover:bg-surface-raised hover:text-contrast disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw
            className={`size-4${isFetching ? " animate-spin" : ""}`}
            aria-hidden
          />
        </button>
      </div>
      {isLoading && (
        <p className="mb-2 text-sm text-muted" aria-live="polite">
          {t(I18nKey.SETTINGS$RUNNING_VERSIONS_CHECKING)}
        </p>
      )}
      <dl className="divide-y divide-border rounded-lg border border-border bg-base-secondary px-4">
        {rows.map((row) => (
          <VersionRow key={row.name} {...row} />
        ))}
      </dl>
    </section>
  );
}
