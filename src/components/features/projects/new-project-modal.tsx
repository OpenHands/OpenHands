import React from "react";
import { v4 as uuidv4 } from "uuid";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import ProjectsService, {
  type Project,
} from "#/api/projects-service/projects-service.api";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { projectsQueryRoot } from "#/hooks/query/use-projects";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import {
  formControlFieldClassName,
  formControlMultilineFieldClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { resolveNewConversationWorkspace } from "#/api/conversation-workspace";

interface NewProjectModalProps {
  onClose: () => void;
  onCreated: (project: Project) => void;
}

/**
 * New project: a name, a goal and an optional agent profile for the
 * conversations the coordinator starts. The Agent Server creates the
 * project's coordinator conversation with it.
 */
export function NewProjectModal({ onClose, onCreated }: NewProjectModalProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: profiles } = useAgentProfiles();
  const [name, setName] = React.useState("");
  const [goal, setGoal] = React.useState("");
  const [workerProfileId, setWorkerProfileId] = React.useState("");
  const [workspace, setWorkspace] = React.useState("");
  const { data: workspaces } = useLocalWorkspaces();

  const { mutate: create, isPending } = useMutation({
    mutationFn: async () => {
      // The coordinator is a normal conversation, so it needs a workspace:
      // the one picked, or a new folder like any new conversation gets.
      const { workingDir } = await resolveNewConversationWorkspace({
        conversationId: uuidv4(),
        workingDir: workspace || undefined,
      });
      return ProjectsService.createProject({
        name: name.trim(),
        description: goal.trim(),
        coordinator: { working_dir: workingDir, title: name.trim() },
        ...(workerProfileId
          ? { worker_agent_profile_id: workerProfileId }
          : {}),
      });
    },
    meta: { disableToast: true },
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) });
      onCreated(project);
    },
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$CREATE_FAILED)),
  });

  const canCreate = name.trim().length > 0 && !isPending;

  return (
    <ModalBackdrop onClose={onClose} aria-label={t(I18nKey.PROJECTS$NEW)}>
      <ModalBody
        width="md"
        testID="new-project-modal"
        className="items-stretch"
      >
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (canCreate) create();
          }}
        >
          <h2 className="text-lg font-medium text-foreground">
            {t(I18nKey.PROJECTS$NEW)}
          </h2>
          <label className="flex flex-col gap-1.5 text-sm text-foreground">
            {t(I18nKey.PROJECTS$NAME)}
            <input
              data-testid="new-project-name"
              className={formControlFieldClassName}
              value={name}
              maxLength={256}
              required
              // eslint-disable-next-line jsx-a11y/no-autofocus
              autoFocus
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm text-foreground">
            {t(I18nKey.PROJECTS$GOAL)}
            <textarea
              data-testid="new-project-goal"
              className={cn(
                formControlMultilineFieldClassName,
                "min-h-24 px-3 py-2 text-sm text-contrast",
              )}
              value={goal}
              maxLength={4096}
              placeholder={t(I18nKey.PROJECTS$GOAL_PLACEHOLDER)}
              onChange={(event) => setGoal(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm text-foreground">
            {t(I18nKey.PROJECTS$WORKER_PROFILE)}
            <select
              data-testid="new-project-worker-profile"
              className={formControlFieldClassName}
              value={workerProfileId}
              onChange={(event) => setWorkerProfileId(event.target.value)}
            >
              <option value="">
                {t(I18nKey.PROJECTS$WORKER_PROFILE_DEFAULT)}
              </option>
              {(profiles?.profiles ?? []).flatMap((profile) =>
                profile.id
                  ? [
                      <option key={profile.id} value={profile.id}>
                        {profile.name}
                      </option>,
                    ]
                  : [],
              )}
            </select>
            <span className="text-xs text-muted">
              {t(I18nKey.PROJECTS$WORKER_PROFILE_HINT)}
            </span>
          </label>
          <label className="flex flex-col gap-1.5 text-sm text-foreground">
            {t(I18nKey.PROJECTS$WORKSPACE)}
            <select
              data-testid="new-project-workspace"
              className={formControlFieldClassName}
              value={workspace}
              onChange={(event) => setWorkspace(event.target.value)}
            >
              <option value="">{t(I18nKey.PROJECTS$WORKSPACE_NEW)}</option>
              {(workspaces?.workspaces ?? []).map((w) => (
                <option key={w.path} value={w.path} title={w.path}>
                  {w.name}
                </option>
              ))}
            </select>
          </label>
          <p className="text-xs text-muted">
            {t(I18nKey.PROJECTS$COORDINATOR_CREATED_HINT)}
          </p>
          <div className="flex justify-end gap-2">
            <BrandButton type="button" variant="secondary" onClick={onClose}>
              {t(I18nKey.BUTTON$CANCEL)}
            </BrandButton>
            <BrandButton
              testId="new-project-create"
              type="submit"
              variant="primary"
              isDisabled={!canCreate}
              aria-busy={isPending}
            >
              {t(I18nKey.PROJECTS$CREATE)}
            </BrandButton>
          </div>
        </form>
      </ModalBody>
    </ModalBackdrop>
  );
}
