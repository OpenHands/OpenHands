import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ProfileRow } from "./profile-row";
import { ProviderConnectionGroup } from "./provider-connection-group";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
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
   * Provider connections to group profiles under. Undefined when the backend
   * cannot address connections (a cloud backend with no org bound), which
   * renders the flat profile list.
   */
  connections?: ProviderConnection[];
  onActivate: (name: string) => void;
  onEdit: (profile: ProfileInfo) => void;
  onRename: (profile: ProfileInfo) => void;
  onDuplicate: (profile: ProfileInfo) => void;
  onDelete: (profile: ProfileInfo) => void;
  onAddModels?: (connection: ProviderConnection) => void;
  onEditConnection?: (connection: ProviderConnection) => void;
  onDeleteConnection?: (connection: ProviderConnection) => void;
  isActivating: boolean;
  isModelUnlisted?: (model: string | null | undefined) => boolean;
}

export type ProfileGroup =
  | {
      kind: "connection";
      connection: ProviderConnection;
      profiles: ProfileInfo[];
    }
  | {
      /** Linked to a connection id that is not in the loaded list. */
      kind: "unknown-connection";
      connectionId: string;
      profiles: ProfileInfo[];
    }
  | { kind: "unlinked"; profiles: ProfileInfo[] };

/**
 * Bucket profiles under their provider connection. Every connection gets a
 * group, in list order, even with no profiles so it can offer "Add models".
 * Profiles referencing an id missing from the list (e.g. the list failed to
 * load) get a group per id, in order of first appearance. Profiles with no
 * connection, or whose connection the backend reports as deleted, collect in a
 * trailing "unlinked" group, omitted when empty.
 */
export function groupProfilesByConnection(
  profiles: ProfileInfo[],
  connections: ProviderConnection[],
): ProfileGroup[] {
  const byConnection = new Map<string, ProfileInfo[]>(
    connections.map((connection) => [connection.id, []]),
  );
  const unknown = new Map<string, ProfileInfo[]>();
  const unlinked: ProfileInfo[] = [];

  for (const profile of profiles) {
    const connectionId = profile.provider_connection_id ?? null;
    if (!connectionId || profile.provider_connection_broken) {
      unlinked.push(profile);
    } else if (byConnection.has(connectionId)) {
      byConnection.get(connectionId)!.push(profile);
    } else {
      const group = unknown.get(connectionId) ?? [];
      group.push(profile);
      unknown.set(connectionId, group);
    }
  }

  const groups: ProfileGroup[] = [
    ...connections.map((connection) => ({
      kind: "connection" as const,
      connection,
      profiles: byConnection.get(connection.id)!,
    })),
    ...[...unknown].map(([connectionId, group]) => ({
      kind: "unknown-connection" as const,
      connectionId,
      profiles: group,
    })),
  ];
  return unlinked.length > 0
    ? [...groups, { kind: "unlinked", profiles: unlinked }]
    : groups;
}

export function ProfilesBody({
  isLoading,
  loadError,
  profiles,
  active,
  canManage,
  connections,
  onActivate,
  onEdit,
  onRename,
  onDuplicate,
  onDelete,
  onAddModels,
  onEditConnection,
  onDeleteConnection,
  isActivating,
  isModelUnlisted,
}: ProfilesBodyProps) {
  const { t } = useTranslation("openhands");

  const renderRow = (profile: ProfileInfo, className?: string) => (
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
      isModelUnlisted={isModelUnlisted?.(profile.model)}
      className={className}
    />
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
        <p className="text-sm text-feedback-error">
          {t(I18nKey.SETTINGS$PROFILES_LOAD_ERROR)}
        </p>
      </div>
    );
  }

  const connectionList = connections ?? [];
  if (profiles.length === 0 && connectionList.length === 0) {
    return (
      <div
        data-testid="profiles-empty"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-muted">
          {t(I18nKey.SETTINGS$PROFILES_EMPTY)}
        </p>
      </div>
    );
  }

  // Group only when connections are addressable and there is at least one
  // connection or linked profile; otherwise the flat list says it all.
  const hasLinkedProfiles = profiles.some((p) => p.provider_connection_id);
  if (!connections || (connectionList.length === 0 && !hasLinkedProfiles)) {
    return (
      <div
        className={cn(
          settingsListContainerClassName,
          settingsListDividerClassName,
        )}
      >
        {profiles.map((profile) => renderRow(profile))}
      </div>
    );
  }

  // Indent rows past the group header's chevron so they read as its children.
  const nestedRowClassName = "pl-10";

  return (
    <div className="flex flex-col gap-3">
      {groupProfilesByConnection(profiles, connectionList).map((group) => {
        const rows = group.profiles.map((profile) =>
          renderRow(profile, nestedRowClassName),
        );
        if (group.kind === "connection") {
          return (
            <ProviderConnectionGroup
              key={group.connection.id}
              label={group.connection.display_name}
              description={group.connection.provider}
              profileCount={group.profiles.length}
              connection={group.connection}
              canManage={canManage}
              onAddModels={onAddModels}
              onEdit={onEditConnection}
              onDelete={onDeleteConnection}
            >
              {rows}
            </ProviderConnectionGroup>
          );
        }
        if (group.kind === "unknown-connection") {
          return (
            <ProviderConnectionGroup
              key={group.connectionId}
              label={group.connectionId}
              profileCount={group.profiles.length}
              connection={null}
              canManage={canManage}
            >
              {rows}
            </ProviderConnectionGroup>
          );
        }
        return (
          <ProviderConnectionGroup
            key="__unlinked__"
            label={t(I18nKey.SETTINGS$PROFILES_UNGROUPED)}
            description={t(I18nKey.SETTINGS$PROFILES_UNGROUPED_DESCRIPTION)}
            profileCount={group.profiles.length}
            connection={null}
            canManage={canManage}
            isUnlinked
          >
            {rows}
          </ProviderConnectionGroup>
        );
      })}
    </div>
  );
}
