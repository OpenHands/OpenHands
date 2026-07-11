import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { CreateApiKeyResponse } from "#/api/api-keys";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { mutateWithToast } from "#/utils/mutate-with-toast";
import { ApiKeyModalBase } from "./api-key-modal-base";
import { useCreateApiKey } from "#/hooks/mutation/use-create-api-key";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { Dropdown } from "#/ui/dropdown/dropdown";
import { DropdownOption } from "#/ui/dropdown/types";

interface CreateApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onKeyCreated: (newKey: CreateApiKeyResponse) => void;
}

// Sentinel value used in the dropdown UI to represent an unbound
// ("All orgs") selection. Distinct from any real org id and from
// ``null`` / ``undefined`` to keep the explicit-vs-omitted distinction
// intact on the wire.
const ALL_ORGS_VALUE = "__all_orgs__";

// Converts a `datetime-local` input value ("2026-01-31T14:30") to a UTC ISO
// string. Returns undefined when the input is empty.
const localDateTimeToIso = (value: string): string | undefined => {
  if (!value) return undefined;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return undefined;
  return date.toISOString();
};

export function CreateApiKeyModal({
  isOpen,
  onClose,
  onKeyCreated,
}: CreateApiKeyModalProps) {
  const { t } = useTranslation();
  const [newKeyName, setNewKeyName] = useState("");
  const [notBefore, setNotBefore] = useState("");
  const [expiresAt, setExpiresAt] = useState("");
  const [selectedOrg, setSelectedOrg] = useState<string>(ALL_ORGS_VALUE);

  const createApiKeyMutation = useCreateApiKey();
  const { data: organizationsData, isLoading: isLoadingOrganizations } =
    useOrganizations();
  const { organizationId: currentOrgId } = useSelectedOrganizationId();

  const orgOptions: DropdownOption[] = useMemo(() => {
    const orgs = organizationsData?.organizations ?? [];
    return [
      // "All orgs" maps to an unbound API key.
      {
        value: ALL_ORGS_VALUE,
        label: t(I18nKey.SETTINGS$API_KEY_ALL_ORGS),
      },
      ...orgs.map((org) => ({
        value: org.id,
        label: org.is_personal ? t(I18nKey.ORG$PERSONAL_WORKSPACE) : org.name,
      })),
    ];
  }, [organizationsData, t]);

  const selectedOption = useMemo(
    () =>
      orgOptions.find((option) => option.value === selectedOrg) ??
      orgOptions[0],
    [orgOptions, selectedOrg],
  );

  const handleCreateKey = async () => {
    if (!newKeyName.trim()) {
      displayErrorToast(t(I18nKey.ERROR$REQUIRED_FIELD));
      return;
    }

    const notBeforeIso = localDateTimeToIso(notBefore);
    const expiresAtIso = localDateTimeToIso(expiresAt);

    if (
      notBeforeIso &&
      expiresAtIso &&
      new Date(notBeforeIso) >= new Date(expiresAtIso)
    ) {
      displayErrorToast(t(I18nKey.SETTINGS$API_KEY_WINDOW_INVALID));
      return;
    }

    // ``null`` (explicit) creates an unbound key. The chosen org id maps
    // directly to a bound key. We forward ``org_id`` only when the user
    // has made an explicit choice so omitting the field falls back to the
    // server's effective-org default.
    const orgIdForApi = selectedOrg === ALL_ORGS_VALUE ? null : selectedOrg;

    const newKey = await mutateWithToast(
      createApiKeyMutation,
      {
        name: newKeyName.trim(),
        not_before: notBeforeIso,
        expires_at: expiresAtIso,
        org_id: orgIdForApi,
      },
      {
        success: t(I18nKey.SETTINGS$API_KEY_CREATED),
        error: t(I18nKey.ERROR$GENERIC),
      },
    ).catch(() => null);

    if (newKey) {
      onKeyCreated(newKey);
      setNewKeyName("");
      setNotBefore("");
      setExpiresAt("");
      // Reset to the user's currently selected org on success so the next
      // open of the modal does not surprise the user with the last choice.
      setSelectedOrg(currentOrgId ?? ALL_ORGS_VALUE);
    }
  };

  const handleCancel = () => {
    setNewKeyName("");
    setNotBefore("");
    setExpiresAt("");
    setSelectedOrg(currentOrgId ?? ALL_ORGS_VALUE);
    onClose();
  };

  const modalFooter = (
    <>
      <BrandButton
        type="button"
        variant="primary"
        className="grow"
        onClick={handleCreateKey}
        isDisabled={
          createApiKeyMutation.isPending ||
          isLoadingOrganizations ||
          !newKeyName.trim()
        }
      >
        {createApiKeyMutation.isPending ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.BUTTON$CREATE)
        )}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={handleCancel}
        isDisabled={createApiKeyMutation.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={t(I18nKey.SETTINGS$CREATE_API_KEY)}
      footer={modalFooter}
    >
      <div data-testid="create-api-key-modal">
        <p className="text-sm text-gray-300">
          {t(I18nKey.SETTINGS$CREATE_API_KEY_DESCRIPTION)}
        </p>
        <SettingsInput
          testId="api-key-name-input"
          label={t(I18nKey.SETTINGS$NAME)}
          placeholder={t(I18nKey.SETTINGS$API_KEY_NAME_PLACEHOLDER)}
          value={newKeyName}
          onChange={(value) => setNewKeyName(value)}
          className="w-full mt-4"
          type="text"
        />
        <div className="mt-6">
          <label
            htmlFor="api-key-org-selector"
            className="text-sm text-gray-300 mb-2 block"
          >
            {t(I18nKey.SETTINGS$API_KEY_ORG_LABEL)}
          </label>
          <Dropdown
            testId="api-key-org-selector"
            defaultValue={selectedOption}
            options={orgOptions}
            loading={isLoadingOrganizations}
            placeholder={t(I18nKey.SETTINGS$API_KEY_ORG_PLACEHOLDER)}
            onChange={(item) => {
              if (item) {
                setSelectedOrg(item.value);
              }
            }}
          />
          <p className="text-xs text-gray-400 mt-2">
            {t(I18nKey.SETTINGS$API_KEY_ORG_HELP)}
          </p>
        </div>
        <p className="text-sm text-gray-300 mt-6">
          {t(I18nKey.SETTINGS$API_KEY_ACTIVE_WINDOW)}
        </p>
        <div className="flex flex-col gap-4 mt-2">
          <SettingsInput
            testId="api-key-not-before-input"
            label={t(I18nKey.SETTINGS$API_KEY_NOT_BEFORE)}
            value={notBefore}
            onChange={(value) => setNotBefore(value)}
            className="w-full"
            type="datetime-local"
            showOptionalTag
          />
          <SettingsInput
            testId="api-key-expires-at-input"
            label={t(I18nKey.SETTINGS$API_KEY_EXPIRES_AT)}
            value={expiresAt}
            onChange={(value) => setExpiresAt(value)}
            className="w-full"
            type="datetime-local"
            showOptionalTag
          />
        </div>
      </div>
    </ApiKeyModalBase>
  );
}
