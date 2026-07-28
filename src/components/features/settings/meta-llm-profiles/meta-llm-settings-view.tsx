import { useState } from "react";
import { useTranslation } from "react-i18next";
import { HttpError } from "@openhands/typescript-client";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useMetaProfiles } from "#/hooks/query/use-meta-profiles";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useSaveMetaProfile } from "#/hooks/mutation/use-save-meta-profile";
import { useActivateMetaProfile } from "#/hooks/mutation/use-activate-meta-profile";
import MetaProfilesService, {
  type MetaProfile,
} from "#/api/meta-profiles-service/meta-profiles-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { MetaProfileEditor } from "./meta-profile-editor";
import { MetaProfileRow } from "./meta-profile-row";
import { DeleteMetaProfileModal } from "./delete-meta-profile-modal";

type ViewMode = "list" | "create" | "edit";

interface EditingMetaProfile {
  name: string;
  config: MetaProfile;
}

export function MetaLlmSettingsView() {
  const { t } = useTranslation("openhands");
  const { data, isLoading, error } = useMetaProfiles();
  const { data: llmProfilesData } = useLlmProfiles();
  const saveMetaProfile = useSaveMetaProfile();
  const activateMetaProfile = useActivateMetaProfile();

  const [view, setView] = useState<ViewMode>("list");
  const [editing, setEditing] = useState<EditingMetaProfile | null>(null);
  const [nameToDelete, setNameToDelete] = useState<string | null>(null);

  const metaProfiles = data?.meta_profiles ?? [];
  const active = data?.active_meta_profile ?? null;
  const availableProfiles = (llmProfilesData?.profiles ?? []).map(
    (p) => p.name,
  );
  const existingNames = metaProfiles.map((p) => p.name);
  // A 404 means the backend predates the /api/meta-profiles endpoints
  // (software-agent-sdk #3744). Surface that explicitly instead of a generic
  // error so the page isn't a dead end on older backends.
  const isUnsupportedBackend =
    error instanceof HttpError && error.status === 404;

  const handleActivate = async (name: string) => {
    try {
      await activateMetaProfile.mutateAsync(name);
      displaySuccessToast(t(I18nKey.SETTINGS$META_PROFILE_ACTIVATED, { name }));
    } catch (activateError) {
      const message =
        activateError instanceof Error
          ? activateError.message
          : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  const handleEdit = async (name: string) => {
    try {
      const detail = await MetaProfilesService.getMetaProfile(name);
      setEditing({ name: detail.name, config: detail.config });
      setView("edit");
    } catch (loadError) {
      const message =
        loadError instanceof Error
          ? loadError.message
          : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  const handleSave = async (name: string, config: MetaProfile) => {
    try {
      await saveMetaProfile.mutateAsync({ name, config });
      displaySuccessToast(t(I18nKey.SETTINGS$META_PROFILE_SAVED, { name }));
      setView("list");
      setEditing(null);
    } catch (saveError) {
      const message =
        saveError instanceof Error
          ? saveError.message
          : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  const handleCancel = () => {
    setView("list");
    setEditing(null);
  };

  if (isUnsupportedBackend) {
    return (
      <p
        data-testid="meta-profile-unsupported"
        className="text-sm text-[var(--oh-muted)]"
      >
        {t(I18nKey.SETTINGS$META_PROFILE_UNSUPPORTED)}
      </p>
    );
  }

  if (view === "create" || view === "edit") {
    return (
      <MetaProfileEditor
        mode={view === "edit" ? "edit" : "create"}
        initialName={editing?.name}
        initialConfig={editing?.config}
        availableProfiles={availableProfiles}
        existingNames={existingNames}
        isSaving={saveMetaProfile.isPending}
        onSave={handleSave}
        onCancel={handleCancel}
      />
    );
  }

  return (
    <>
      <div className="flex flex-col gap-4">
        {availableProfiles.length === 0 ? (
          <p
            data-testid="meta-profile-no-llm-profiles"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_NO_LLM_PROFILES)}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-medium text-white">
            {t(I18nKey.SETTINGS$META_PROFILES_AVAILABLE)}
          </h2>
          <BrandButton
            testId="add-meta-profile"
            type="button"
            variant="secondary"
            className="ml-auto"
            onClick={() => {
              setEditing(null);
              setView("create");
            }}
          >
            {t(I18nKey.SETTINGS$ADD_META_PROFILE)}
          </BrandButton>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-6">
            <LoadingSpinner size="small" />
          </div>
        ) : null}

        {error ? (
          <p className="text-sm text-red-400">{t(I18nKey.ERROR$GENERIC)}</p>
        ) : null}

        {!isLoading && !error && metaProfiles.length === 0 ? (
          <p
            data-testid="meta-profile-empty"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$META_PROFILE_NO_PROFILES)}
          </p>
        ) : null}

        {metaProfiles.length > 0 ? (
          <div className="flex flex-col gap-2" data-testid="meta-profile-list">
            {metaProfiles.map((info) => (
              <MetaProfileRow
                key={info.name}
                info={info}
                isActive={info.name === active}
                onActivate={handleActivate}
                onEdit={handleEdit}
                onDelete={setNameToDelete}
                isActivating={activateMetaProfile.isPending}
              />
            ))}
          </div>
        ) : null}
      </div>

      <DeleteMetaProfileModal
        name={nameToDelete}
        onClose={() => setNameToDelete(null)}
      />
    </>
  );
}
