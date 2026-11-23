import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation } from "@tanstack/react-query";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { BrandButton } from "#/components/features/settings/brand-button";
import ProjectsService from "#/api/projects-service/projects-service.api";
import { useProjectArtifacts } from "#/hooks/query/use-projects";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";
import { I18nKey } from "#/i18n/declaration";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formatTimeDelta } from "#/utils/format-time-delta";

/**
 * Picks one of the project's files and adds it to the composer like a file
 * from disk: an image is embedded in the message, any other file is uploaded
 * to the conversation's workspace when the message is sent.
 */
export function ProjectFilePicker({
  projectId,
  onClose,
}: {
  projectId: string;
  onClose: () => void;
}) {
  const { t } = useTranslation("openhands");
  const {
    data: artifacts,
    isLoading,
    isError,
  } = useProjectArtifacts(projectId);
  const { handleUpload } = useChatAttachmentUpload();

  const attach = useMutation({
    mutationFn: async (name: string) => {
      const blob = await ProjectsService.downloadArtifact(projectId, name);
      await handleUpload([new File([blob], name, { type: blob.type })]);
    },
    meta: { disableToast: true },
    onSuccess: onClose,
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$ATTACH_FAILED)),
  });

  const sorted = [...(artifacts ?? [])].sort((a, b) =>
    b.updated_at.localeCompare(a.updated_at),
  );

  let body: React.ReactNode;
  if (isLoading) {
    body = <p className="text-sm text-muted">{t(I18nKey.PROJECTS$LOADING)}</p>;
  } else if (isError) {
    body = (
      <p className="text-sm text-danger">
        {t(I18nKey.PROJECTS$FILES_LOAD_FAILED)}
      </p>
    );
  } else if (!sorted.length) {
    body = (
      <p className="text-sm text-muted">{t(I18nKey.PROJECTS$FILES_EMPTY)}</p>
    );
  } else {
    body = (
      <ul className="flex max-h-80 flex-col divide-y divide-border overflow-auto rounded-lg border border-control">
        {sorted.map((artifact) => (
          <li key={artifact.name}>
            <button
              type="button"
              data-testid="project-file-pick"
              className="flex w-full items-baseline justify-between gap-3 px-3 py-2 text-left text-sm text-foreground hover:bg-contrast/10 disabled:opacity-50"
              disabled={attach.isPending}
              aria-busy={attach.isPending && attach.variables === artifact.name}
              onClick={() => attach.mutate(artifact.name)}
            >
              <span className="truncate font-medium">{artifact.name}</span>
              <span className="shrink-0 text-xs text-muted">
                {formatTimeDelta(artifact.updated_at)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <ModalBackdrop
      onClose={attach.isPending ? undefined : onClose}
      aria-label={t(I18nKey.PROJECTS$ATTACH_TITLE)}
    >
      <div
        data-testid="project-file-picker"
        className="flex w-112 max-w-full flex-col gap-3 rounded-xl border border-border bg-base-secondary p-4"
      >
        <h2 className="text-base font-medium text-foreground">
          {t(I18nKey.PROJECTS$ATTACH_TITLE)}
        </h2>
        {body}
        <div className="flex justify-end">
          <BrandButton
            testId="project-file-picker-cancel"
            type="button"
            variant="secondary"
            isDisabled={attach.isPending}
            onClick={onClose}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
