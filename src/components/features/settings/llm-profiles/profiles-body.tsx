import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ProfileRow } from "./profile-row";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  settingsListContainerClassName,
  settingsListDividerClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";

interface ProfilesBodyProps {
  isLoading: boolean;
  loadError: Error | null;
  profiles: ProfileInfo[];
  active: string | null;
  /** When false, rows render read-only (no actions menu) — cloud members. */
  canManage: boolean;
  onActivate: (name: string) => void;
  onEdit: (profile: ProfileInfo) => void;
  onRename: (profile: ProfileInfo) => void;
  onDuplicate: (profile: ProfileInfo) => void;
  onDelete: (profile: ProfileInfo) => void;
  isActivating: boolean;
}

export function ProfilesBody({
  isLoading,
  loadError,
  profiles,
  active,
  canManage,
  onActivate,
  onEdit,
  onRename,
  onDuplicate,
  onDelete,
  isActivating,
}: ProfilesBodyProps) {
  const { t } = useTranslation("openhands");

  if (isLoading) {
    return (
      <div className="flex justify-center p-4">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (loadError) {
    return (
      <div
        data-testid="profiles-load-error"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-red-400">
          {t(I18nKey.SETTINGS$PROFILES_LOAD_ERROR)}
        </p>
      </div>
    );
  }

  if (profiles.length === 0) {
    return (
      <div
        data-testid="profiles-empty"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$PROFILES_EMPTY)}
        </p>
      </div>
    );
  }

  return (
    <div
      className={cn(
        settingsListContainerClassName,
        settingsListDividerClassName,
      )}
    >
      {profiles.map((profile) => (
        <ProfileRow
          key={profile.name}
          profile={profile}
          isActive={profile.name === active}
          canManage={canManage}
          onActivate={onActivate}
          onEdit={onEdit}
          onRename={onRename}
          onDuplicate={onDuplicate}
          onDelete={onDelete}
          isActivating={isActivating}
        />
      ))}
    </div>
  );
}
