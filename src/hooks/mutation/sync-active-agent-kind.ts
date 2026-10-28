import type { QueryClient } from "@tanstack/react-query";
import type { BackendKind } from "#/api/backend-registry/types";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import { buildAgentSettingsDiffForProfile } from "#/constants/acp-providers";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Keep the global `agent_settings.agent_kind` in step with the agent profile
 * the user last chose (#18883).
 *
 * Activation is pointer-only by design and onboarding is the only writer of
 * the global kind, so after an ACP onboarding and a switch back to an
 * OpenHands profile the cloud account's `agent_settings` still said `acp` —
 * and the home switcher plus profile-less launches followed that stale value.
 * These helpers write the active profile's kind through the same
 * `agent_settings_diff` settings write onboarding uses. Cloud only: local
 * activation keeps its no-settings-write behavior. A failed sync surfaces
 * through the existing settings-save error handling and must never undo the
 * activation or save it follows; the next activation or active-profile save
 * syncs the kind.
 */

/**
 * The backend and org a sync chain was started for. Every call in the chain
 * resolves its destination from the *current* selection when it issues the
 * request, so a chain that spans an await has to prove the selection it read
 * the profile from is still the one it is about to write to.
 */
interface SelectionIdentity {
  backendId: string;
  orgId: string | null;
}

function captureSelectionIdentity(): SelectionIdentity {
  const active = getActiveBackend();
  return { backendId: active.backend.id, orgId: active.orgId };
}

function isCurrentSelection(identity: SelectionIdentity): boolean {
  const active = getActiveBackend();
  return (
    active.backend.id === identity.backendId && active.orgId === identity.orgId
  );
}

function surfaceSyncError(error: unknown): void {
  displayErrorToast(
    retrieveAxiosErrorMessage(error) || i18n.t(I18nKey.ERROR$GENERIC),
  );
}

/**
 * Write the stored profile's kind into `agent_settings`. Saving the active
 * profile (kind changed or not) is deliberately an idempotent re-state of the
 * profile's own fields — the sync self-heals a previously stale account on
 * the next activation or save without a backfill migration.
 */
async function writeAgentSettingsKindSync(
  queryClient: QueryClient,
  name: string,
  identity: SelectionIdentity,
): Promise<void> {
  if (!isCurrentSelection(identity)) return;

  let diff: Record<string, unknown>;
  try {
    const detail = await AgentProfilesService.getProfile(name);
    diff = buildAgentSettingsDiffForProfile(detail.profile);
  } catch (error) {
    surfaceSyncError(error);
    return;
  }

  // `getProfile` and `saveSettings` each resolve the active backend when they
  // issue their request, so the read and the write can land on different
  // accounts when the user switches backend or org mid-chain. Re-check before
  // writing rather than mirroring one account's kind onto another.
  if (!isCurrentSelection(identity)) return;

  try {
    await SettingsService.saveSettings({ agent_settings_diff: diff });
  } catch (error) {
    surfaceSyncError(error);
    return;
  }

  SettingsService.invalidateCache();
  await queryClient.invalidateQueries({
    queryKey: SETTINGS_QUERY_KEYS.personal(),
  });
}

/**
 * After a profile activation resolves, sync the activated profile's kind.
 * Activation carries only the profile UUID, so the summary (name) comes from
 * a fresh list call routed to the active backend.
 */
export async function syncAgentSettingsKindAfterProfileActivation(
  queryClient: QueryClient,
  profileId: string,
  backendKind: BackendKind,
): Promise<void> {
  if (backendKind !== "cloud") return;

  // The caller reads the kind from a render-time snapshot, which can already
  // be stale by the time the activation settles.
  const identity = captureSelectionIdentity();
  if (!isCurrentSelection(identity)) return;

  let list: Awaited<ReturnType<typeof AgentProfilesService.listProfiles>>;
  try {
    list = await AgentProfilesService.listProfiles();
  } catch (error) {
    // The lookup belongs to the sync, not to the activation: the activation
    // has already succeeded, so report the failure and let the next
    // activation or save correct the stored kind.
    surfaceSyncError(error);
    return;
  }
  if (!isCurrentSelection(identity)) return;

  const summary = list.profiles.find((p) => p.id === profileId) ?? null;
  if (!summary?.name) return;

  await writeAgentSettingsKindSync(queryClient, summary.name, identity);
}

/**
 * After a profile save resolves, sync the saved profile's kind when it is the
 * currently active one. A list call is the pre-invalidation source for the
 * active pointer; the summary's `agent_kind` is post-save and cannot detect a
 * kind change, so every active-profile save re-states the kind instead.
 */
export async function syncAgentSettingsKindAfterProfileSave(
  queryClient: QueryClient,
  name: string,
  backendKind: BackendKind,
): Promise<void> {
  if (backendKind !== "cloud") return;

  const identity = captureSelectionIdentity();
  if (!isCurrentSelection(identity)) return;

  let list: Awaited<ReturnType<typeof AgentProfilesService.listProfiles>>;
  try {
    list = await AgentProfilesService.listProfiles();
  } catch (error) {
    surfaceSyncError(error);
    return;
  }
  if (!isCurrentSelection(identity)) return;

  const summary = list.profiles.find((p) => p.name === name);
  if (summary?.id == null) return;
  if (summary.id !== (list.active_agent_profile_id ?? null)) return;

  await writeAgentSettingsKindSync(queryClient, name, identity);
}
