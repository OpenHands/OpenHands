import React from "react";
import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import TerminalIcon from "#/icons/terminal.svg?react";
import { HighlightedSourceView } from "#/components/features/files-tab/highlighted-source-view";
import { useAutomationTarballFiles } from "#/hooks/query/use-automation-tarball-files";
import type { Automation } from "#/types/automation";
import type { TarEntry } from "#/utils/tar-unpack";
import { cn } from "#/utils/utils";
import { SectionCard } from "./section-card";

interface ScriptSectionProps {
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
 * What a script automation runs: the files of its uploaded bundle, read
 * back from the same tarball the kebab menu offers for download. Takes the
 * place of the prompt section for automations that have no prompt.
 */
export function ScriptSection({ automation }: ScriptSectionProps) {
  const { t } = useTranslation("openhands");
  const {
    data: entries,
    isPending,
    isError,
  } = useAutomationTarballFiles(automation);

  const [expandedPaths, setExpandedPaths] = React.useState<Set<string>>(
    () => new Set(),
  );

  const togglePath = (path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const entrypointPath = findEntrypointPath(
    automation.entrypoint,
    entries ?? [],
  );
  // A bundle the service can't hand back (external URL, deleted upload) or
  // that isn't an archive gets the same answer: download it instead.
  const unavailable = isError || entries?.length === 0;

  return (
    <SectionCard
      icon={<TerminalIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT)}
    >
      <div className="flex flex-col gap-4">
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
          ? orderEntries(entries, entrypointPath).map((entry) => {
              const isExpanded = expandedPaths.has(entry.path);
              return (
                <div
                  key={entry.path}
                  data-testid="automation-script-file"
                  className="overflow-hidden rounded-lg border border-[var(--oh-border)]"
                >
                  <button
                    type="button"
                    data-testid="automation-script-file-toggle"
                    aria-expanded={isExpanded}
                    aria-label={entry.path}
                    onClick={() => togglePath(entry.path)}
                    className={cn(
                      "flex w-full items-center justify-between gap-2 bg-black/20 px-3 py-2 font-mono text-xs text-content transition-colors hover:bg-black/30 cursor-pointer text-left",
                      isExpanded && "border-b border-[var(--oh-border)]",
                    )}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <ChevronRight
                        className={cn(
                          "size-3.5 shrink-0 transition-transform text-muted",
                          isExpanded && "rotate-90",
                        )}
                        aria-hidden
                      />
                      <span className="truncate">{entry.path}</span>
                      {entry.path === entrypointPath ? (
                        <span className="shrink-0 rounded-md bg-surface-raised px-2 py-0.5 text-[10px] font-medium text-muted">
                          {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_ENTRYPOINT)}
                        </span>
                      ) : null}
                    </div>
                  </button>
                  {isExpanded &&
                    (entry.text === null ? (
                      <p className="px-3 py-2 text-xs italic text-muted">
                        {t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_BINARY_FILE)}
                      </p>
                    ) : (
                      <div className="max-h-96 overflow-auto">
                        <HighlightedSourceView
                          path={entry.path}
                          text={entry.text}
                        />
                      </div>
                    ))}
                </div>
              );
            })
          : null}
      </div>
    </SectionCard>
  );
}
