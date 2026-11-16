import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronRight } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { HighlightedSourceView } from "#/components/features/files-tab/highlighted-source-view";
import { useAutomationTarballFiles } from "#/hooks/query/use-automation-tarball-files";
import type { Automation } from "#/types/automation";
import type { TarEntry } from "#/utils/tar-unpack";
import { cn } from "#/utils/utils";

interface ScriptBodyProps {
  automation: Automation;
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

/**
 * macOS tar adds AppleDouble (`._name`) and `__MACOSX/` metadata members and
 * Finder's `.DS_Store`; none of them are part of what the script runs.
 */
function isArchiveMetadata(path: string): boolean {
  const segments = path.split("/");
  const name = segments[segments.length - 1];
  return (
    segments[0] === "__MACOSX" || name.startsWith("._") || name === ".DS_Store"
  );
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

function ScriptFile({
  entry,
  isEntrypoint,
}: {
  entry: TarEntry;
  isEntrypoint: boolean;
}) {
  const { t } = useTranslation("openhands");
  const [isOpen, setIsOpen] = useState(false);
  const isBinary = entry.text === null;

  return (
    <div
      data-testid="automation-script-file"
      className="overflow-hidden rounded-lg border border-border"
    >
      <button
        type="button"
        aria-expanded={isBinary ? undefined : isOpen}
        disabled={isBinary}
        onClick={() => setIsOpen((open) => !open)}
        className="flex w-full items-center gap-2 bg-surface-deep px-3 py-2 text-left font-mono text-xs text-content enabled:hover:bg-surface-raised dark:bg-black/20"
      >
        <ChevronRight
          className={cn(
            "size-3.5 shrink-0 text-muted transition-transform",
            isOpen && "rotate-90",
            isBinary && "invisible",
          )}
          aria-hidden="true"
        />
        <span className="truncate">{entry.path}</span>
        {isEntrypoint ? (
          <span className="shrink-0 rounded-md bg-surface-raised px-2 py-0.5 text-3xs font-medium text-muted">
            {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_ENTRYPOINT)}
          </span>
        ) : null}
        {isBinary ? (
          <span className="ml-auto shrink-0 font-sans italic text-muted">
            {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_BINARY_FILE)}
          </span>
        ) : null}
      </button>
      {isOpen && entry.text !== null ? (
        <div className="max-h-96 overflow-auto border-t border-border">
          <HighlightedSourceView path={entry.path} text={entry.text} />
        </div>
      ) : null}
    </div>
  );
}

/**
 * What a script automation runs: its entrypoint and the files of its uploaded
 * bundle, read back from the same tarball the kebab menu offers for
 * download. Each file opens to its source on demand.
 */
export function ScriptBody({ automation }: ScriptBodyProps) {
  const { t } = useTranslation("openhands");
  const {
    data: allEntries,
    isPending,
    isError,
  } = useAutomationTarballFiles(automation);
  const entries = allEntries?.filter((entry) => !isArchiveMetadata(entry.path));

  const entrypointPath = findEntrypointPath(
    automation.entrypoint,
    entries ?? [],
  );
  // A bundle the service can't hand back (external URL, deleted upload) or
  // that isn't an archive gets the same answer: download it instead.
  const unavailable = isError || entries?.length === 0;

  return (
    <div className="flex flex-col gap-3">
      {automation.entrypoint ? (
        <code
          data-testid="automation-script-entrypoint"
          className="self-start rounded-md bg-surface-raised px-2 py-1 font-mono text-xs text-content"
        >
          {automation.entrypoint}
        </code>
      ) : null}

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
            <ScriptFile
              key={entry.path}
              entry={entry}
              isEntrypoint={entry.path === entrypointPath}
            />
          ))
        : null}
    </div>
  );
}
