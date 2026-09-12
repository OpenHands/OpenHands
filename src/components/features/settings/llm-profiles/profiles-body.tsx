import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ProfileRow } from "./profile-row";
import { BrandBadge } from "#/components/shared/badge";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  isOracleProfileName,
  ORACLE_PROFILE_NAME,
} from "#/utils/oracle-profile";
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
   * visually clustered. Empty (the default, e.g. a cloud backend with no org
   * bound) renders a flat list identical to before.
   */
  connectionNamesById?: Record<string, string>;
  onActivate: (name: string) => void;
  onEdit: (profile: ProfileInfo) => void;
  onRename: (profile: ProfileInfo) => void;
  onDuplicate: (profile: ProfileInfo) => void;
  onDelete: (profile: ProfileInfo) => void;
  onConfigureOracle?: () => void;
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

export function shouldShowOracleProfileEntry(profiles: ProfileInfo[]): boolean {
  return !profiles.some((profile) => isOracleProfileName(profile.name));
}

function OracleProfileEntry({
  canManage,
  onConfigure,
}: {
  canManage: boolean;
  onConfigure: () => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="oracle-profile-entry"
      aria-disabled={!canManage}
      className="flex items-center justify-between gap-4 rounded-lg border border-dashed border-[var(--oh-border-subtle)] bg-[var(--oh-surface-secondary)] p-4 opacity-80"
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="text-sm font-medium text-[var(--oh-foreground)]"
            title={ORACLE_PROFILE_NAME}
          >
            {ORACLE_PROFILE_NAME}
          </span>
          <BrandBadge className="px-2 py-0.5 text-xs">
            {t(I18nKey.SETTINGS$ORACLE_PROFILE_BADGE)}
          </BrandBadge>
        </div>
        <p
          data-testid="oracle-profile-description"
          className="mt-1 max-w-2xl text-sm leading-5 text-[var(--oh-muted)]"
        >
          {t(I18nKey.SETTINGS$ORACLE_PROFILE_DESCRIPTION)}
        </p>
      </div>
      {canManage ? (
        <BrandButton
          testId="configure-oracle-profile"
          type="button"
          variant="secondary"
          className="shrink-0"
          onClick={onConfigure}
        >
          {t(I18nKey.SETTINGS$ORACLE_PROFILE_CONFIGURE)}
        </BrandButton>
      ) : (
        <span className="shrink-0 text-xs text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$ORACLE_PROFILE_VIEW_ONLY)}
        </span>
      )}
    </div>
  );
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
  onConfigureOracle = () => undefined,
  isActivating,
}: ProfilesBodyProps) {
  const { t } = useTranslation("openhands");
  const oracleEntry = shouldShowOracleProfileEntry(profiles) ? (
    <OracleProfileEntry canManage={canManage} onConfigure={onConfigureOracle} />
  ) : null;

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
      isReserved={isOracleProfileName(profile.name)}
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
      <div className="flex flex-col gap-3">
        {oracleEntry}
        <div
          data-testid="profiles-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.SETTINGS$PROFILES_EMPTY)}
          </p>
        </div>
      </div>
    );
  }

  // Group only when there is at least one linked connection to show; otherwise
  // (no profile links to a connection) render the flat list unchanged.
  const hasLinkedProfiles = profiles.some((p) => p.provider_connection_id);
  if (!hasLinkedProfiles) {
    return (
      <div className="flex flex-col gap-3">
        {oracleEntry}
        <div className={listClassName}>{profiles.map(renderRow)}</div>
      </div>
    );
  }

  const groups = groupProfilesByConnection(profiles, connectionNamesById);
  return (
    <div className="flex flex-col gap-4">
      {oracleEntry}
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
