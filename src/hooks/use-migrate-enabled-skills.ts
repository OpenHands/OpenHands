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
 * Move a workspace to an explicit `enabled_skills` allow-list, once.
 *
 * It runs at the app root, not on the Customize page: until it has run the
 * resolver falls back to the curated default, which would silently drop
 * catalog skills an existing workspace had switched on. Local only — cloud
 * never reads the field.
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

    migratedBackendRef.current = backend.id;
    const migrated = migrateSkillEnablement(toSkillEnablement(settings));
    // Silent on failure: the resolver's fallback keeps the session working and
    // the next app load retries.
    if (migrated) saveSettings(migrated);
  }, [isLocal, isLoading, isError, settings, backend.id, saveSettings]);
}
