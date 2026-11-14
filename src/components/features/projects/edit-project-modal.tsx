import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import ProjectsService, {
  type Project,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { projectsQueryRoot } from "#/hooks/query/use-projects";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import {
  formControlFieldClassName,
  formControlMultilineFieldClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";

interface EditProjectModalProps {
  project: Project;
  onClose: () => void;
  onDeleted: () => void;
}

/** Rename a project, change its goal, or delete it. */
export function EditProjectModal({
  project,
  onClose,
  onDeleted,
}: EditProjectModalProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const [name, setName] = React.useState(project.name);
  const [goal, setGoal] = React.useState(project.description);
  const [isConfirmingDelete, setIsConfirmingDelete] = React.useState(false);
  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) });

  const save = useMutation({
    mutationFn: () =>
      ProjectsService.updateProject(project.id, {
        name: name.trim(),
        description: goal.trim(),
      }),
    meta: { disableToast: true },
    onSuccess: () => {
      refresh();
      onClose();
    },
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPDATE_FAILED)),
  });
  const remove = useMutation({
    mutationFn: () => ProjectsService.deleteProject(project.id),
    meta: { disableToast: true },
    onSuccess: () => {
      refresh();
      onDeleted();
    },
    onError: (error) => {
      setIsConfirmingDelete(false);
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPDATE_FAILED));
    },
  });

  const busy = save.isPending || remove.isPending;
  const canSave = name.trim().length > 0 && !busy;

  if (isConfirmingDelete) {
    return (
      <ConfirmationModal
        text={t(I18nKey.PROJECTS$DELETE_CONFIRM, { name: project.name })}
        confirmText={t(I18nKey.BUTTON$DELETE)}
        isConfirming={remove.isPending}
        onCancel={() => setIsConfirmingDelete(false)}
        onConfirm={() => remove.mutate()}
      />
    );
  }

  return (
    <ModalBackdrop onClose={onClose} aria-label={t(I18nKey.PROJECTS$EDIT)}>
      <ModalBody
        width="md"
        testID="edit-project-modal"
        className="items-stretch"
      >
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (canSave) save.mutate();
          }}
        >
          <h2 className="text-lg font-medium text-foreground">
            {t(I18nKey.PROJECTS$EDIT)}
          </h2>
          <label className="flex flex-col gap-1.5 text-sm text-foreground">
            {t(I18nKey.PROJECTS$NAME)}
            <input
              data-testid="edit-project-name"
              className={formControlFieldClassName}
              value={name}
              maxLength={256}
              required
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm text-foreground">
            {t(I18nKey.PROJECTS$GOAL)}
            <textarea
              data-testid="edit-project-goal"
              className={cn(
                formControlMultilineFieldClassName,
                "min-h-24 px-3 py-2 text-sm text-contrast",
              )}
              value={goal}
              maxLength={4096}
              onChange={(event) => setGoal(event.target.value)}
            />
          </label>
          <div className="flex flex-wrap justify-between gap-2">
            <BrandButton
              testId="edit-project-delete"
              type="button"
              variant="danger"
              isDisabled={busy}
              onClick={() => setIsConfirmingDelete(true)}
            >
              {t(I18nKey.PROJECTS$DELETE)}
            </BrandButton>
            <span className="flex gap-2">
              <BrandButton type="button" variant="secondary" onClick={onClose}>
                {t(I18nKey.BUTTON$CANCEL)}
              </BrandButton>
              <BrandButton
                testId="edit-project-save"
                type="submit"
                variant="primary"
                isDisabled={!canSave}
                aria-busy={save.isPending}
              >
                {t(I18nKey.BUTTON$SAVE)}
              </BrandButton>
            </span>
          </div>
        </form>
      </ModalBody>
    </ModalBackdrop>
  );
}
