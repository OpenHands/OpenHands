import { useQuery } from "@tanstack/react-query";
import {
  getMarsBridge,
  isOpenHandsConfig,
  type MarsAgentConfig,
  type MarsAuthState,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import { MARS_QUERY_KEYS } from "./query-keys";

const PAGE_SIZE = 50;
/** Sessions pause, wake, and provision on their own; keep the list honest. */
const REFETCH_INTERVAL_MS = 10_000;

export interface MarsAgentGroup {
  config: MarsAgentConfig;
  sessions: MarsSession[];
}

export interface MarsAgentsSnapshot {
  groups: MarsAgentGroup[];
  /** OpenHands-config sessions missing from their config's own listing. */
  otherSessions: MarsSession[];
}

const DESTROYED = "SESSION_STATUS_DESTROYED";

function byRecentActivity(a: MarsSession, b: MarsSession): number {
  const at = (s: MarsSession) =>
    Date.parse(s.last_event_at ?? s.created_at ?? "") || 0;
  return at(b) - at(a);
}

/** Destroyed sessions are gone for good; listing them is only noise. */
function visibleSessions(sessions: MarsSession[]): MarsSession[] {
  return sessions.filter((s) => s.status !== DESTROYED).sort(byRecentActivity);
}

async function fetchMarsAgents(): Promise<MarsAgentsSnapshot> {
  const bridge = getMarsBridge();
  if (!bridge) return { groups: [], otherSessions: [] };

  const [{ configs }, { sessions: teamSessions }] = await Promise.all([
    bridge.listAgentConfigs({ pageSize: PAGE_SIZE }),
    bridge.listSessions({ pageSize: PAGE_SIZE }),
  ]);

  const openHandsConfigs = configs.filter(isOpenHandsConfig);
  const openHandsConfigIds = new Set(openHandsConfigs.map((c) => c.id));

  const groups = await Promise.all(
    openHandsConfigs.map(async (config) => {
      const { sessions } = await bridge.listConfigSessions(config.id, {
        pageSize: PAGE_SIZE,
      });
      return { config, sessions: visibleSessions(sessions) };
    }),
  );

  const grouped = new Set(
    groups.flatMap((group) => group.sessions.map((s) => s.session_id)),
  );
  return {
    groups,
    otherSessions: visibleSessions(
      teamSessions.filter(
        (s) =>
          !grouped.has(s.session_id) &&
          openHandsConfigIds.has(s.config_id ?? ""),
      ),
    ),
  };
}

export function useMarsAgents(
  connectionId: string | null,
  {
    refetchIntervalMs = REFETCH_INTERVAL_MS,
  }: { refetchIntervalMs?: number } = {},
) {
  return useQuery({
    queryKey: MARS_QUERY_KEYS.agents(connectionId),
    queryFn: fetchMarsAgents,
    enabled: connectionId !== null && getMarsBridge() !== null,
    refetchInterval: refetchIntervalMs,
    retry: false,
    meta: { disableToast: true },
  });
}

/** Stored DigitalOcean connections; tokens never leave the main process. */
export function useMarsAuthState() {
  const bridge = getMarsBridge();
  return useQuery({
    queryKey: MARS_QUERY_KEYS.authState,
    queryFn: () => bridge!.getAuthState(),
    enabled: bridge !== null,
    retry: false,
    meta: { disableToast: true },
  });
}

/** The active connection's id while it is still usable, else null. */
export function getUsableConnectionId(
  authState: MarsAuthState | undefined,
): string | null {
  const active = authState?.active;
  return active && !active.isExpired ? active.id : null;
}
