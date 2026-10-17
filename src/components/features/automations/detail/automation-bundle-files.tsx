import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { HighlightedSourceView } from "#/components/features/files-tab/highlighted-source-view";
import { useAutomationTarballFiles } from "#/hooks/query/use-automation-tarball-files";
import type { Automation } from "#/types/automation";
import type { TarEntry } from "#/utils/tar-unpack";

interface AutomationBundleFilesProps {
  automation: Pick<Automation, "id" | "updated_at" | "entrypoint">;
}

/**
 * The bundle member the entrypoint runs, if it names one: the first token
 * of the command that is a file in the archive (`python main.py` → `main.py`).
 */
function findEntrypointPath(
  entrypoint: string | undefined,
  entries: TarEntry[],
): string | null {
  if (!entrypoint) return null;
  const paths = new Set(entries.map((entry) => entry.path));
  for (const token of entrypoint.split(/\s+/)) {
    const candidate = token.replace(/^\.\//, "");
    if (paths.has(candidate)) return candidate;
  }
  return null;
}

/** Entrypoint first, then the rest by path. */
function orderEntries(
  entries: TarEntry[],
  entrypointPath: string | null,
): TarEntry[] {
  return [...entries].sort((a, b) => {
    if (a.path === entrypointPath) return -1;
    if (b.path === entrypointPath) return 1;
    return a.path.localeCompare(b.path);
  });
}

/**
 * Read-only view of the files in an automation's uploaded bundle, read back
 * from the same tarball the kebab menu offers for download. Shared by the
 * detail page and the edit form so both show what the automation runs.
 */
export function AutomationBundleFiles({
  automation,
}: AutomationBundleFilesProps) {
  const { t } = useTranslation("openhands");
  const {
    data: entries,
    isPending,
    isError,
  } = useAutomationTarballFiles(automation);

  const entrypointPath = findEntrypointPath(
    automation.entrypoint,
    entries ?? [],
  );
  // A bundle the service can't hand back (external URL, deleted upload) or
  // that isn't an archive gets the same answer: download it instead.
  const unavailable = isError || entries?.length === 0;

  return (
    <>
      {isPending ? (
        <p className="text-sm italic text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_LOADING)}
        </p>
      ) : null}

      {unavailable ? (
        <p
          data-testid="automation-script-unavailable"
          className="text-sm text-muted"
        >
          {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_UNAVAILABLE)}
        </p>
      ) : null}

      {entries && entries.length > 0
        ? orderEntries(entries, entrypointPath).map((entry) => (
            <div
              key={entry.path}
              data-testid="automation-script-file"
              className="overflow-hidden rounded-lg border border-[var(--oh-border)]"
            >
              <div className="flex items-center gap-2 border-b border-[var(--oh-border)] bg-black/20 px-3 py-2 font-mono text-xs text-content">
                <span className="truncate">{entry.path}</span>
                {entry.path === entrypointPath ? (
                  <span className="shrink-0 rounded-md bg-surface-raised px-2 py-0.5 text-[10px] font-medium text-muted">
                    {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_ENTRYPOINT)}
                  </span>
                ) : null}
              </div>
              {entry.text === null ? (
                <p className="px-3 py-2 text-xs italic text-muted">
                  {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_BINARY_FILE)}
                </p>
              ) : (
                <div className="max-h-96 overflow-auto">
                  <HighlightedSourceView path={entry.path} text={entry.text} />
                </div>
              )}
            </div>
          ))
        : null}
    </>
  );
}
