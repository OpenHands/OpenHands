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
  /**
   * Display name per provider-connection id. When non-empty, profiles are
   * grouped under their connection's name so models sharing a provider are
   * visually clustered. Empty (the default, and always on cloud) renders a flat
   * list identical to before.
   */
  connectionNamesById?: Record<string, string>;
  onActivate: (name: string) => void;
  onEdit: (profile: ProfileInfo) => void;
  onRename: (profile: ProfileInfo) => void;
  onDuplicate: (profile: ProfileInfo) => void;
  onDelete: (profile: ProfileInfo) => void;
  isActivating: boolean;
}

interface ProfileGroup {
  /** Connection id, or null for profiles with no provider connection. */
  connectionId: string | null;
  label: string | null;
  profiles: ProfileInfo[];
}

/**
 * Bucket profiles by their `provider_connection_id`, preserving input order
 * within each group and ordering groups by first appearance. Unlinked profiles
 * collect under a trailing `null` group.
 */
export function groupProfilesByConnection(
  profiles: ProfileInfo[],
  connectionNamesById: Record<string, string>,
): ProfileGroup[] {
  const groups = new Map<string, ProfileGroup>();
  const unlinked: ProfileGroup = {
    connectionId: null,
    label: null,
    profiles: [],
  };

  for (const profile of profiles) {
    const connectionId = profile.provider_connection_id ?? null;
    if (!connectionId) {
      unlinked.profiles.push(profile);
      continue;
    }
    let group = groups.get(connectionId);
    if (!group) {
      group = {
        connectionId,
        label: connectionNamesById[connectionId] ?? connectionId,
        profiles: [],
      };
      groups.set(connectionId, group);
    }
    group.profiles.push(profile);
  }

  const linkedGroups = [...groups.values()];
  return unlinked.profiles.length > 0
    ? [...linkedGroups, unlinked]
    : linkedGroups;
}

export function ProfilesBody({
  isLoading,
  loadError,
  profiles,
  active,
  canManage,
  connectionNamesById = {},
  onActivate,
  onEdit,
  onRename,
  onDuplicate,
  onDelete,
  isActivating,
}: ProfilesBodyProps) {
  const { t } = useTranslation("openhands");

  const renderRow = (profile: ProfileInfo) => (
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
  );

  const listClassName = cn(
    settingsListContainerClassName,
    settingsListDividerClassName,
  );

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

  // Group only when there is at least one linked connection to show; otherwise
  // (every profile today, and always on cloud) render the flat list unchanged.
  const hasLinkedProfiles = profiles.some((p) => p.provider_connection_id);
  if (!hasLinkedProfiles) {
    return <div className={listClassName}>{profiles.map(renderRow)}</div>;
  }

  const groups = groupProfilesByConnection(profiles, connectionNamesById);
  return (
    <div className="flex flex-col gap-4">
      {groups.map((group) => (
        <div
          key={group.connectionId ?? "__unlinked__"}
          className="flex flex-col gap-2"
        >
          <h3
            data-testid="profile-group-header"
            className="text-xs font-medium uppercase tracking-wide text-[var(--oh-muted)]"
          >
            {group.label ?? t(I18nKey.SETTINGS$PROFILES_UNGROUPED)}
          </h3>
          <div className={listClassName}>{group.profiles.map(renderRow)}</div>
        </div>
      ))}
    </div>
  );
}
