import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  useProjectMemory,
  useProjectMemoryFile,
} from "#/hooks/query/use-projects";
import { formatTimeDelta } from "#/utils/format-time-delta";

function MemoryFile({
  projectId,
  name,
  revision,
}: {
  projectId: string;
  name: string;
  revision: string;
}) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, isError } = useProjectMemoryFile(
    projectId,
    name,
    revision,
  );
  let content = data ?? "";
  if (isLoading) content = t(I18nKey.PROJECTS$LOADING);
  else if (isError) content = t(I18nKey.PROJECTS$FILE_PREVIEW_FAILED);
  return (
    <pre
      data-testid="project-memory-preview"
      className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-control bg-canvas-base px-3 py-2 text-xs text-foreground"
    >
      {content}
    </pre>
  );
}

/** What the project's conversations remember: MEMORY.md and its topic files. */
export function ProjectMemory({ projectId }: { projectId: string }) {
  const { t } = useTranslation("openhands");
  const { data: files } = useProjectMemory(projectId);
  const [open, setOpen] = React.useState<string | null>(null);
  const openFile = (files ?? []).find((file) => file.name === open) ?? null;

  return (
    <section
      data-testid="project-memory"
      aria-labelledby="project-memory-title"
      className="flex flex-col gap-3"
    >
      <h2
        id="project-memory-title"
        className="text-base font-medium text-foreground"
      >
        {t(I18nKey.PROJECTS$MEMORY)} · {files?.length ?? 0}
      </h2>
      {!files?.length ? (
        <p className="text-sm text-muted">{t(I18nKey.PROJECTS$MEMORY_EMPTY)}</p>
      ) : (
        <ul className="flex flex-col divide-y divide-border rounded-lg border border-control">
          {files.map((file) => (
            <li
              key={file.name}
              data-testid="project-memory-file"
              className="flex flex-wrap items-baseline justify-between gap-x-3 px-3 py-2"
            >
              <button
                type="button"
                className="w-fit text-left text-sm font-medium text-foreground hover:underline"
                aria-expanded={open === file.name}
                onClick={() =>
                  setOpen((current) =>
                    current === file.name ? null : file.name,
                  )
                }
              >
                {file.name}
              </button>
              <span className="text-xs text-muted">
                {formatTimeDelta(file.updated_at)}
              </span>
            </li>
          ))}
        </ul>
      )}
      {openFile ? (
        <MemoryFile
          projectId={projectId}
          name={openFile.name}
          revision={openFile.revision}
        />
      ) : null}
    </section>
  );
}
