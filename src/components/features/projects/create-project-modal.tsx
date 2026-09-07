import React from "react";
import { useTranslation } from "react-i18next";
import type { CreateProjectPayload } from "#/api/projects-service/projects-types";
import { DEFAULT_PROJECT_BRANCH } from "#/api/projects-service/projects-constants";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { I18nKey } from "#/i18n/declaration";

export interface CreateProjectModalProps {
  isOpen: boolean;
  isPending?: boolean;
  onClose: () => void;
  onSubmit: (payload: CreateProjectPayload) => void;
}

export function CreateProjectModal({
  isOpen,
  isPending,
  onClose,
  onSubmit,
}: CreateProjectModalProps) {
  const { t } = useTranslation("openhands");
  const [name, setName] = React.useState("");
  const [repoUrl, setRepoUrl] = React.useState("");
  const [localPath, setLocalPath] = React.useState("");
  const [defaultBranch, setDefaultBranch] = React.useState(
    DEFAULT_PROJECT_BRANCH,
  );
  const [costCap, setCostCap] = React.useState("");

  if (!isOpen) return null;

  const footer = (
    <>
      <BrandButton
        type="button"
        variant="tertiary"
        testId="create-project-cancel"
        onClick={onClose}
        isDisabled={isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        type="button"
        variant="primary"
        testId="create-project-submit"
        isDisabled={isPending || !name.trim()}
        onClick={() => {
          onSubmit({
            name: name.trim(),
            repo_url: repoUrl.trim() || null,
            local_path: localPath.trim() || null,
            default_branch: defaultBranch.trim() || DEFAULT_PROJECT_BRANCH,
            cost_cap: costCap.trim() ? Number(costCap) : null,
          });
        }}
      >
        {t(I18nKey.PROJECTS$CREATE)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={t(I18nKey.PROJECTS$CREATE_TITLE)}
      footer={footer}
      onClose={onClose}
      width="md"
    >
      <div
        className="flex w-full flex-col gap-3"
        data-testid="create-project-form"
      >
        <SettingsInput
          testId="create-project-name"
          name="project-name"
          label={t(I18nKey.PROJECTS$NAME)}
          type="text"
          value={name}
          onChange={setName}
          showRequiredTag
        />
        <SettingsInput
          testId="create-project-repo-url"
          name="repo-url"
          label={t(I18nKey.PROJECTS$REPO_URL)}
          type="text"
          value={repoUrl}
          onChange={setRepoUrl}
          showOptionalTag
        />
        <SettingsInput
          testId="create-project-local-path"
          name="local-path"
          label={t(I18nKey.PROJECTS$LOCAL_PATH)}
          type="text"
          value={localPath}
          onChange={setLocalPath}
          showOptionalTag
        />
        <SettingsInput
          testId="create-project-branch"
          name="default-branch"
          label={t(I18nKey.PROJECTS$DEFAULT_BRANCH)}
          type="text"
          value={defaultBranch}
          onChange={setDefaultBranch}
        />
        <SettingsInput
          testId="create-project-cost-cap"
          name="cost-cap"
          label={t(I18nKey.PROJECTS$COST_CAP)}
          type="number"
          value={costCap}
          onChange={setCostCap}
          showOptionalTag
        />
      </div>
    </ApiKeyModalBase>
  );
}
