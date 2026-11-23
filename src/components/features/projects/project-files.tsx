import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Download, Trash2, Upload } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import ProjectsService, {
  projectIdKey,
  type ProjectArtifact,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  projectsQueryRoot,
  useProjectArtifactBlob,
  useProjectArtifacts,
  useProjectArtifactText,
} from "#/hooks/query/use-projects";
import { projectArtifactKind } from "#/utils/project-artifact-kind";
import { downloadBlob } from "#/utils/utils";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formatTimeDelta } from "#/utils/format-time-delta";

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

const PREVIEW_BOX =
  "max-h-96 overflow-auto rounded-lg border border-control bg-canvas-base";

function TextPreview({
  projectId,
  artifact,
}: {
  projectId: string;
  artifact: ProjectArtifact;
}) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, isError } = useProjectArtifactText(
    projectId,
    artifact.name,
    artifact.updated_at,
  );
  let content: string;
  if (isLoading) content = t(I18nKey.PROJECTS$LOADING);
  else if (isError) content = t(I18nKey.PROJECTS$FILE_PREVIEW_FAILED);
  else content = data ?? "";
  return (
    <pre
      data-testid="project-file-preview"
      className={`${PREVIEW_BOX} whitespace-pre-wrap break-words px-3 py-2 text-xs text-foreground`}
    >
      {content}
    </pre>
  );
}

/** An object URL for the blob, revoked when it changes or unmounts. */
function useObjectUrl(blob: Blob | undefined) {
  const [url, setUrl] = React.useState<string | null>(null);
  React.useEffect(() => {
    if (!blob) {
      setUrl(null);
      return undefined;
    }
    const next = URL.createObjectURL(blob);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob]);
  return url;
}

/**
 * Images and PDFs, from a blob Canvas re-typed by extension. An image renders
 * in an <img>, where an SVG's scripts never run; a PDF is typed
 * `application/pdf`, so the browser's viewer opens it and nothing else can.
 */
function MediaPreview({
  projectId,
  artifact,
  kind,
}: {
  projectId: string;
  artifact: ProjectArtifact;
  kind: "image" | "pdf";
}) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, isError } = useProjectArtifactBlob(
    projectId,
    artifact.name,
    artifact.updated_at,
  );
  const url = useObjectUrl(data);
  if (isError || (!isLoading && !url)) {
    return (
      <p data-testid="project-file-preview" className="text-sm text-danger">
        {t(I18nKey.PROJECTS$FILE_PREVIEW_FAILED)}
      </p>
    );
  }
  if (!url) {
    return (
      <p data-testid="project-file-preview" className="text-sm text-muted">
        {t(I18nKey.PROJECTS$LOADING)}
      </p>
    );
  }
  if (kind === "image") {
    return (
      <div
        data-testid="project-file-preview"
        data-kind="image"
        className={`${PREVIEW_BOX} flex justify-center p-2`}
      >
        <img
          src={url}
          alt={artifact.name}
          className="max-h-88 max-w-full object-contain"
        />
      </div>
    );
  }
  return (
    <iframe
      data-testid="project-file-preview"
      data-kind="pdf"
      src={url}
      title={artifact.name}
      className="h-128 w-full rounded-lg border border-control bg-canvas-base"
    />
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
  const kind = projectArtifactKind(artifact.name);
  if (kind === "text") {
    return <TextPreview projectId={projectId} artifact={artifact} />;
  }
  if (kind === "image" || kind === "pdf") {
    return (
      <MediaPreview
        key={`${artifact.name}:${artifact.updated_at}`}
        projectId={projectId}
        artifact={artifact}
        kind={kind}
      />
    );
  }
  return (
    <p data-testid="project-file-preview" className="text-sm text-muted">
      {t(I18nKey.PROJECTS$FILE_NO_PREVIEW)}
    </p>
  );
}

const ICON_BUTTON =
  "rounded-md p-1.5 text-muted hover:bg-contrast/10 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50";

/** The project's files: who added each, upload, preview, download, delete. */
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

  const download = useMutation({
    mutationFn: async (name: string) => {
      downloadBlob(
        await ProjectsService.downloadArtifact(projectId, name),
        name,
      );
    },
    meta: { disableToast: true },
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$FILE_DOWNLOAD_FAILED)),
  });

  const [confirmingDelete, setConfirmingDelete] = React.useState<string | null>(
    null,
  );
  const remove = useMutation({
    mutationFn: (name: string) =>
      ProjectsService.deleteArtifact(projectId, name),
    meta: { disableToast: true },
    onSuccess: (_, name) =>
      setOpen((current) => (current === name ? null : current)),
    onSettled: () => {
      setConfirmingDelete(null);
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) });
    },
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$FILE_DELETE_FAILED)),
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
              className="flex items-center gap-2 px-3 py-2"
            >
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
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
              </div>
              <button
                type="button"
                data-testid="project-file-download"
                className={ICON_BUTTON}
                aria-label={t(I18nKey.PROJECTS$FILE_DOWNLOAD, {
                  name: artifact.name,
                })}
                title={t(I18nKey.PROJECTS$FILE_DOWNLOAD, {
                  name: artifact.name,
                })}
                disabled={
                  download.isPending && download.variables === artifact.name
                }
                onClick={() => download.mutate(artifact.name)}
              >
                <Download width={16} height={16} aria-hidden />
              </button>
              <button
                type="button"
                data-testid="project-file-delete"
                className={ICON_BUTTON}
                aria-label={t(I18nKey.PROJECTS$FILE_DELETE, {
                  name: artifact.name,
                })}
                title={t(I18nKey.PROJECTS$FILE_DELETE, {
                  name: artifact.name,
                })}
                onClick={() => setConfirmingDelete(artifact.name)}
              >
                <Trash2 width={16} height={16} aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {openArtifact ? (
        <Preview projectId={projectId} artifact={openArtifact} />
      ) : null}
      {confirmingDelete ? (
        <ConfirmationModal
          text={t(I18nKey.PROJECTS$FILE_DELETE_CONFIRM, {
            name: confirmingDelete,
          })}
          confirmText={t(I18nKey.BUTTON$DELETE)}
          isConfirming={remove.isPending}
          onCancel={() => setConfirmingDelete(null)}
          onConfirm={() => remove.mutate(confirmingDelete)}
        />
      ) : null}
    </section>
  );
}
