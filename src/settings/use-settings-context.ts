import { useActiveBackend } from "#/contexts/active-backend-context";
import type { SettingsContext } from "./registry";

/**
 * Derive the host-owned {@link SettingsContext} that settings sections gate
 * their visibility on. Today that is just the active backend kind; new facts
 * (capabilities, feature flags, role/permission) are added here so every
 * section sees a single, consistent source of truth.
 */
export function useSettingsContext(): SettingsContext {
  const { backend } = useActiveBackend();
  return { backendKind: backend.kind };
}
