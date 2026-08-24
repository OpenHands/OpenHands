import { useEffect, useRef } from "react";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import {
  migrateSkillEnablement,
  toSkillEnablement,
} from "#/utils/skill-enablement";

/**
 * Convert a workspace from the old "every catalog skill on, minus a deny-list"
 * default to an explicit `enabled_skills` allow-list, once, on first load.
 *
 * It runs at the app root rather than on the Customize page because the
 * conversion has to happen for users who never open that page: until it does,
 * `resolveEnabledCatalogSkills` falls back to the curated default, which would
 * silently drop catalog skills an existing workspace had switched on.
 *
 * Local backends only. Cloud creates conversations through its own server-side
 * catalog and never reads `enabled_skills`, so writing one there would persist
 * a preference nothing acts on.
 */
export function useMigrateEnabledSkills(): void {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local" && !isNoBackend(backend);
  const { data: settings, isLoading, isError } = useSettings();
  const { mutate: saveSettings } = useSaveSettings();

  // One attempt per backend: the save invalidates the settings query, so
  // without this the refetch would re-enter before the write is visible.
  const migratedBackendRef = useRef<string | null>(null);

  useEffect(() => {
    migratedBackendRef.current = null;
  }, [backend.id]);

  useEffect(() => {
    if (!isLocal || isLoading || isError || !settings) return;
    if (migratedBackendRef.current === backend.id) return;

    const migrated = migrateSkillEnablement(toSkillEnablement(settings));
    if (!migrated) {
      migratedBackendRef.current = backend.id;
      return;
    }

    migratedBackendRef.current = backend.id;
    // A failure here is silent on purpose: the resolver's fallback keeps the
    // session working, and the next app load retries.
    saveSettings(migrated);
  }, [isLocal, isLoading, isError, settings, backend.id, saveSettings]);
}
