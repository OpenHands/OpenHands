import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  sessionActivityKey,
  useSessionActivityStore,
  type SessionActivity,
} from "#/stores/session-activity-store";
const EMPTY: SessionActivity[] = [];
export function useSessionActivities(
  conversationId: string | null | undefined,
) {
  const { backend, orgId } = useActiveBackend();
  const key = sessionActivityKey(backend.id, orgId, conversationId ?? "");
  const sessions = useSessionActivityStore(
    (state) => state.entries[key] ?? EMPTY,
  );
  return { key, sessions };
}
