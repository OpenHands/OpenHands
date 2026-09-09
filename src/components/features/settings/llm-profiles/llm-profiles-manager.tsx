import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { RenameProfileModal } from "./rename-profile-modal";
import { DeleteProfileModal } from "./delete-profile-modal";
import { ProfilesBody } from "./profiles-body";
import { ProviderConnectionsManager } from "./provider-connections-manager";
import { SubscriptionModelsBody } from "./subscription-models-body";
import ProfilesService, {
  ProfileInfo,
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import { ROUTING_PATH } from "#/api/routing-service/routing-constants";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useProviderConnections } from "#/hooks/query/use-provider-connections";
import { useSubscriptionModelCatalog } from "#/hooks/query/use-subscription-model-catalog";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import { useSubscriptionModelEnablement } from "#/hooks/use-subscription-model-enablement";
import { useSyncChatgptSubscriptionProfiles } from "#/hooks/use-sync-chatgpt-subscription-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { isChatgptAutoProfileName } from "#/utils/subscription-model-catalog";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";

interface LlmProfilesManagerProps {
  onAddProfile?: () => void;
  onEditProfile?: (profile: ProfileInfo) => void;
}

export function LlmProfilesManager({
  onAddProfile,
  onEditProfile,
}: LlmProfilesManagerProps) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, error } = useLlmProfiles();
  const activateProfile = useActivateLlmProfile();
  const saveProfile = useSaveLlmProfile();
  const canManage = useCanManageOrgProfiles();
  const { backend, orgId } = useActiveBackend();
  const supportsConnections =
    backend.kind === "local" || (backend.kind === "cloud" && !!orgId);
  const {
    data: connections,
    isLoading: isLoadingConnections,
    error: connectionsError,
  } = useProviderConnections();
  const catalog = useSubscriptionModelCatalog();
  const { isOfferEnabled, setOfferEnabled } = useSubscriptionModelEnablement();
  const [profileToRename, setProfileToRename] = useState<ProfileInfo | null>(
    null,
  );
  const [profileToDelete, setProfileToDelete] = useState<ProfileInfo | null>(
    null,
  );

  const profiles = data?.profiles ?? [];
  const customProfiles = useMemo(
    () => profiles.filter((profile) => !isChatgptAutoProfileName(profile.name)),
    [profiles],
  );
  const active = data?.active_profile ?? null;
  const connectionList = useMemo(() => connections ?? [], [connections]);

  useSyncChatgptSubscriptionProfiles(canManage ? catalog.offers : [], profiles);

  const connectionNamesById = useMemo(
    () => Object.fromEntries(connectionList.map((c) => [c.id, c.display_name])),
    [connectionList],
  );
  const linkedCountById = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const profile of customProfiles) {
      const id = profile.provider_connection_id;
      if (id) counts[id] = (counts[id] ?? 0) + 1;
    }
    return counts;
  }, [customProfiles]);

  const handleActivate = async (name: string) => {
    try {
      await activateProfile.mutateAsync(name);
      displaySuccessToast(t(I18nKey.SETTINGS$PROFILE_ACTIVATED, { name }));
    } catch (err) {
      console.error("Failed to activate profile:", err);
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  const handleEdit = (profile: ProfileInfo) => {
    onEditProfile?.(profile);
  };

  const handleDuplicate = async (profile: ProfileInfo) => {
    try {
      const detail = await ProfilesService.getProfile(
        profile.name,
        "encrypted",
      );

      const existingNames = new Set(profiles.map((p) => p.name));
      let newName = `${profile.name}-copy`;
      let counter = 1;
      while (existingNames.has(newName)) {
        newName = `${profile.name}-copy-${counter}`;
        counter += 1;
      }

      await saveProfile.mutateAsync({
        name: newName,
        request: {
          llm: detail.config as SaveProfileRequest["llm"],
          include_secrets: true,
        },
      });

      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_DUPLICATED, { name: newName }),
      );
    } catch (err) {
      console.error("Failed to duplicate profile:", err);
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  const showCatalog = catalog.offers.length > 0;
  const showCustom = customProfiles.length > 0;
  const showEmpty =
    !isLoading && !catalog.isLoading && !error && !showCatalog && !showCustom;

  return (
    <>
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-col gap-1">
              <h2 className="text-base font-medium text-white">
                {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
              </h2>
              <p className="text-sm text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$LLM_PROFILES_SUBLINE)}
              </p>
              <NavigationLink
                to={ROUTING_PATH}
                data-testid="llm-task-routing-link"
                className="w-fit text-sm text-[var(--oh-muted)] underline-offset-2 hover:text-white hover:underline"
              >
                {t(I18nKey.SETTINGS$LLM_TASK_ROUTING)}
              </NavigationLink>
            </div>
            {onAddProfile && canManage ? (
              <BrandButton
                testId="add-llm-profile"
                type="button"
                variant="secondary"
                className="ml-auto"
                onClick={onAddProfile}
              >
                {t(I18nKey.SETTINGS$ADD_LLM_PROFILE)}
              </BrandButton>
            ) : null}
          </div>

          {catalog.isLoading ? (
            <div className="flex justify-center p-4">
              <LoadingSpinner size="large" />
            </div>
          ) : showCatalog ? (
            <SubscriptionModelsBody
              offers={catalog.offers}
              rows={catalog.rows}
              isOfferEnabled={isOfferEnabled}
              onToggle={(offer, enabled) => {
                void setOfferEnabled(offer, enabled);
              }}
              canManage={canManage}
            />
          ) : null}

          {showEmpty ? (
            <div
              data-testid="profiles-empty"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$SUBSCRIPTION_MODELS_EMPTY)}
              </p>
            </div>
          ) : null}

          {showCustom || isLoading || error ? (
            <div className="flex flex-col gap-2">
              {showCustom ? (
                <h3 className="text-xs font-medium uppercase tracking-wide text-[var(--oh-muted)]">
                  {t(I18nKey.SETTINGS$CUSTOM_LLM_PROFILES)}
                </h3>
              ) : null}
              <ProfilesBody
                isLoading={isLoading}
                loadError={error ?? null}
                profiles={customProfiles}
                active={active}
                canManage={canManage}
                connectionNamesById={connectionNamesById}
                onActivate={handleActivate}
                onEdit={handleEdit}
                onRename={setProfileToRename}
                onDuplicate={handleDuplicate}
                onDelete={setProfileToDelete}
                isActivating={activateProfile.isPending}
              />
            </div>
          ) : null}
        </div>

        {supportsConnections && canManage ? (
          <ProviderConnectionsManager
            connections={connectionList}
            linkedCountById={linkedCountById}
            catalogCountsBySource={catalog.countsBySource}
            isLoading={isLoadingConnections}
            loadError={connectionsError ?? null}
          />
        ) : null}
      </div>

      <RenameProfileModal
        profile={profileToRename}
        onClose={() => setProfileToRename(null)}
      />
      <DeleteProfileModal
        profile={profileToDelete}
        onClose={() => setProfileToDelete(null)}
      />
    </>
  );
}
