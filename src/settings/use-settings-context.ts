import { useActiveBackend } from "#/contexts/active-backend-context";
import { useConfig } from "#/hooks/query/use-config";
import type { SettingsContext } from "./registry";

/**
 * Derive the host-owned {@link SettingsContext} that contributed settings
 * surfaces (sections and nav/page entries) gate their visibility on. This is
 * the single place the fact set is assembled, so every consumer sees a
 * consistent source of truth; new facts (role/permission, capabilities) are
 * added here.
 */
export function useSettingsContext(): SettingsContext {
  const { backend, orgId } = useActiveBackend();
  const { data: config } = useConfig();
  return {
    backendKind: backend.kind,
    orgId,
    featureFlags: config?.feature_flags,
  };
}
