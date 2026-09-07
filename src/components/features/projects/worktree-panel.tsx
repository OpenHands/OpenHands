import React from "react";
import { useTranslation } from "react-i18next";
import type { ProjectWorktree } from "#/api/projects-service/projects-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";

const WORKTREE_STATUS_KEY: Record<string, I18nKey> = {
  idle: I18nKey.PROJECTS$STATUS_IDLE,
  working: I18nKey.PROJECTS$WORKTREE_WORKING,
  reviewing: I18nKey.PROJECTS$WORKTREE_REVIEWING,
  ci: I18nKey.PROJECTS$WORKTREE_CI,
  merged: I18nKey.PROJECTS$WORKTREE_MERGED,
  error: I18nKey.PROJECTS$STATUS_ERROR,
};

export interface WorktreePanelProps {
  worktrees: ProjectWorktree[];
  onAdd?: (branchName: string) => void;
  onRemove?: (worktreeId: string) => void;
  onAssign?: (worktreeId: string, agentSessionId: string) => void;
}

export function WorktreePanel({
  worktrees,
  onAdd,
  onRemove,
  onAssign,
}: WorktreePanelProps) {
  const { t } = useTranslation("openhands");
  const [branchName, setBranchName] = React.useState("");
  const [sessionId, setSessionId] = React.useState("");

  return (
    <section data-testid="worktree-panel">
      <h2 className="mb-3 text-sm font-medium text-white">
        {t(I18nKey.PROJECTS$WORKTREES)}
      </h2>
      {onAssign ? (
        <div className="mb-3">
          <SettingsInput
            testId="worktree-session-id"
            name="agent-session"
            label={t(I18nKey.PROJECTS$AGENT_SESSION)}
            type="text"
            value={sessionId}
            onChange={setSessionId}
          />
        </div>
      ) : null}
      {worktrees.length === 0 ? (
        <p
          data-testid="worktree-empty"
          className="mb-3 text-sm text-tertiary-light"
        >
          {t(I18nKey.PROJECTS$NO_WORKTREES)}
        </p>
      ) : (
        <ul className="mb-4 flex flex-col gap-2">
          {worktrees.map((worktree) => (
            <li
              key={worktree.id}
              data-testid={`worktree-row-${worktree.id}`}
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-[rgba(255,255,255,0.04)] p-3"
            >
              <div className="min-w-0">
                <p className="text-sm font-medium text-white">
                  {worktree.branch_name}
                </p>
                <p className="text-xs text-tertiary-light">
                  {t(I18nKey.PROJECTS$ASSIGNED_AGENT)}
                  {worktree.agent_session_id
                    ? `: ${worktree.agent_session_id}`
                    : ""}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span
                  data-testid={`worktree-status-${worktree.id}`}
                  className={extensionModuleCardPillClassName}
                >
                  {t(
                    WORKTREE_STATUS_KEY[worktree.status] ??
                      I18nKey.PROJECTS$STATUS_IDLE,
                  )}
                </span>
                {onAssign ? (
                  <BrandButton
                    type="button"
                    variant="secondary"
                    testId={`worktree-assign-${worktree.id}`}
                    isDisabled={!sessionId.trim()}
                    onClick={() => onAssign(worktree.id, sessionId.trim())}
                  >
                    {t(I18nKey.PROJECTS$ASSIGN_AGENT)}
                  </BrandButton>
                ) : null}
                {onRemove ? (
                  <BrandButton
                    type="button"
                    variant="danger"
                    testId={`worktree-remove-${worktree.id}`}
                    onClick={() => onRemove(worktree.id)}
                  >
                    {t(I18nKey.PROJECTS$REMOVE_WORKTREE)}
                  </BrandButton>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      )}
      {onAdd ? (
        <form
          className="flex flex-col gap-3 sm:flex-row sm:items-end"
          onSubmit={(event) => {
            event.preventDefault();
            if (!branchName.trim()) return;
            onAdd(branchName.trim());
            setBranchName("");
          }}
        >
          <SettingsInput
            testId="worktree-branch-name"
            name="branch-name"
            label={t(I18nKey.PROJECTS$BRANCH_NAME)}
            type="text"
            value={branchName}
            onChange={setBranchName}
          />
          <BrandButton
            type="submit"
            variant="primary"
            testId="worktree-add"
            className="shrink-0"
          >
            {t(I18nKey.PROJECTS$ADD_WORKTREE)}
          </BrandButton>
        </form>
      ) : null}
    </section>
  );
}
