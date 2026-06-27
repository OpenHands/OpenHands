import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ProfileNameInput } from "#/components/features/settings/profile-name-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { useRenameOrgLlmProfile } from "#/hooks/mutation/use-org-llm-profile-mutations";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { mutateWithToast } from "#/utils/mutate-with-toast";
import { extractErrorMessage } from "#/utils/extract-error-message";
import { I18nKey } from "#/i18n/declaration";
import { PROFILE_NAME_PATTERN } from "#/utils/derive-profile-name";

interface OrgRenameProfileModalProps {
  orgId: string;
  profile: LlmProfileSummary | null;
  onClose: () => void;
}

export function OrgRenameProfileModal({
  orgId,
  profile,
  onClose,
}: OrgRenameProfileModalProps) {
  const { t } = useTranslation();
  const [newName, setNewName] = useState("");
  const renameProfile = useRenameOrgLlmProfile(orgId);

  useEffect(() => {
    setNewName(profile?.name ?? "");
  }, [profile?.name]);

  if (!profile) return null;

  const trimmed = newName.trim();
  const isUnchanged = trimmed === profile.name;
  const isValid = PROFILE_NAME_PATTERN.test(trimmed);

  const handleSubmit = async () => {
    if (!isValid) {
      displayErrorToast(t(I18nKey.SETTINGS$PROFILE_NAME_RULE));
      return;
    }
    if (isUnchanged) {
      onClose();
      return;
    }
    const ok = await mutateWithToast(
      renameProfile,
      { name: profile.name, newName: trimmed },
      {
        success: t(I18nKey.SETTINGS$PROFILE_RENAMED, { name: trimmed }),
        error: (err) => extractErrorMessage(err, t(I18nKey.ERROR$GENERIC)),
      },
    ).catch(() => null);
    if (ok !== null) onClose();
  };

  const footer = (
    <>
      <BrandButton
        testId="org-rename-profile-submit"
        type="button"
        variant="primary"
        className="grow"
        onClick={handleSubmit}
        isDisabled={renameProfile.isPending || !isValid}
      >
        {renameProfile.isPending ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.BUTTON$RENAME)
        )}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={onClose}
        isDisabled={renameProfile.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$PROFILE_RENAME_TITLE)}
      footer={footer}
    >
      <div
        data-testid="org-rename-profile-modal"
        className="flex flex-col gap-3"
      >
        <ProfileNameInput
          testId="org-rename-profile-input"
          ruleTestId="org-rename-profile-rule"
          value={newName}
          onChange={setNewName}
        />
      </div>
    </ApiKeyModalBase>
  );
}
