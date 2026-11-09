import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import ProjectsService, {
  projectIdKey,
  type ProjectArtifact,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  projectsQueryRoot,
  useProjectArtifacts,
  useProjectArtifactText,
} from "#/hooks/query/use-projects";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formatTimeDelta } from "#/utils/format-time-delta";

const TEXT_FILE =
  /\.(md|markdown|txt|json|csv|log|ya?ml|toml|diff|patch|ts|tsx|js|py|html|css|sh)$/i;

function Provenance({
  artifact,
  titles,
}: {
  artifact: ProjectArtifact;
  titles: Map<string, string>;
}) {
  const { t } = useTranslation("openhands");
  if (artifact.source === "user") {
    return <span>{t(I18nKey.PROJECTS$FILE_FROM_YOU)}</span>;
  }
  if (!artifact.conversation_id) {
    return <span>{t(I18nKey.PROJECTS$FILE_FROM_A_CONVERSATION)}</span>;
  }
  const id = projectIdKey(artifact.conversation_id);
  return (
    <span>
      {t(I18nKey.PROJECTS$FILE_FROM_CONVERSATION)}{" "}
      <NavigationLink
        to={`/conversations/${id}`}
        className="text-link underline"
        title={id}
      >
        {titles.get(id) ?? id.slice(0, 8)}
      </NavigationLink>
    </span>
  );
}

function Preview({
  projectId,
  artifact,
}: {
  projectId: string;
  artifact: ProjectArtifact;
}) {
  const { t } = useTranslation("openhands");
  const isText = TEXT_FILE.test(artifact.name);
  const { data, isLoading, isError } = useProjectArtifactText(
    projectId,
    isText ? artifact.name : null,
    artifact.updated_at,
  );
  let content: string;
  if (!isText) content = t(I18nKey.PROJECTS$FILE_NO_PREVIEW);
  else if (isLoading) content = t(I18nKey.PROJECTS$LOADING);
  else if (isError) content = t(I18nKey.PROJECTS$FILE_PREVIEW_FAILED);
  else content = data ?? "";
  return (
    <pre
      data-testid="project-file-preview"
      className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-control bg-canvas-base px-3 py-2 text-xs text-foreground"
    >
      {content}
    </pre>
  );
}

/** The project's files: who added each, upload, and a text preview. */
export function ProjectFiles({
  projectId,
  titles,
}: {
  projectId: string;
  titles: Map<string, string>;
}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: artifacts, isError } = useProjectArtifacts(projectId);
  const [open, setOpen] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const upload = useMutation({
    mutationFn: (file: File) => ProjectsService.uploadArtifact(projectId, file),
    meta: { disableToast: true },
    onSuccess: (artifact) => setOpen(artifact.name),
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) }),
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPLOAD_FAILED)),
  });

  const sorted = [...(artifacts ?? [])].sort((a, b) =>
    b.updated_at.localeCompare(a.updated_at),
  );
  const openArtifact = sorted.find((a) => a.name === open) ?? null;

  return (
    <section
      data-testid="project-files"
      aria-labelledby="project-files-title"
      className="flex flex-col gap-3"
    >
      <div className="flex items-center justify-between gap-2">
        <h2
          id="project-files-title"
          className="text-base font-medium text-foreground"
        >
          {t(I18nKey.PROJECTS$FILES)} · {sorted.length}
        </h2>
        <BrandButton
          testId="project-file-upload"
          type="button"
          variant="secondary"
          isDisabled={upload.isPending}
          aria-busy={upload.isPending}
          startContent={<Upload width={14} height={14} />}
          onClick={() => inputRef.current?.click()}
        >
          {t(I18nKey.PROJECTS$UPLOAD)}
        </BrandButton>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          data-testid="project-file-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) upload.mutate(file);
            // eslint-disable-next-line no-param-reassign
            event.target.value = "";
          }}
        />
      </div>
      {isError ? (
        <p className="text-sm text-danger">
          {t(I18nKey.PROJECTS$FILES_LOAD_FAILED)}
        </p>
      ) : null}
      {!isError && !sorted.length ? (
        <p className="text-sm text-muted">{t(I18nKey.PROJECTS$FILES_EMPTY)}</p>
      ) : null}
      {sorted.length ? (
        <ul className="flex flex-col divide-y divide-border rounded-lg border border-control">
          {sorted.map((artifact) => (
            <li
              key={artifact.name}
              data-testid="project-file"
              data-source={artifact.source}
              className="flex flex-col gap-0.5 px-3 py-2"
            >
              <button
                type="button"
                className="w-fit text-left text-sm font-medium text-foreground hover:underline"
                aria-expanded={open === artifact.name}
                onClick={() =>
                  setOpen((current) =>
                    current === artifact.name ? null : artifact.name,
                  )
                }
              >
                {artifact.name}
              </button>
              <span className="flex flex-wrap gap-x-3 text-xs text-muted">
                <Provenance artifact={artifact} titles={titles} />
                <span>{formatTimeDelta(artifact.updated_at)}</span>
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      {openArtifact ? (
        <Preview projectId={projectId} artifact={openArtifact} />
      ) : null}
    </section>
  );
}
