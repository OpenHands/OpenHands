import { useTranslation } from "react-i18next";

import {
  getSessionPhase,
  type MarsSession,
  type MarsSessionPhase,
} from "#/api/mars/mars-tunnel-backend";
import type {
  MarsConnectingState,
  MarsConnectStage,
} from "#/hooks/use-connect-mars-session";
import {
  formatRelativeTime,
  isInvalidTimestamp,
} from "#/utils/format-relative-time";
import {
  getSessionDisplayName,
  PHASE_LABEL_KEY,
  STAGE_LABEL_KEY,
} from "./managed-agents-labels";

/** The stage of the connect in flight for this session, if any. */
export function getConnectStage(
  connecting: MarsConnectingState | null,
  session: MarsSession,
): MarsConnectStage | null {
  return connecting?.sessionId === session.session_id ? connecting.stage : null;
}

/**
 * When the session last did anything, or null when the backend left the
 * timestamp unset (the epoch) or unparseable.
 */
export function getSessionLastActivity(
  session: MarsSession | undefined,
): string | null {
  const at = session?.last_event_at ?? session?.created_at;
  return isInvalidTimestamp(at) ? null : at!;
}

/** "5 minutes ago" for the session's last activity, or null when unknown. */
export function useSessionActivityLabel(
  session: MarsSession | undefined,
): string | null {
  const { t, i18n } = useTranslation("openhands");
  const at = getSessionLastActivity(session);
  return at ? formatRelativeTime(at, i18n.language, t) : null;
}

export interface SessionMeta {
  name: string;
  phase: MarsSessionPhase;
  phaseLabel: string;
  /** Label for the connect stage in flight, or null when not connecting. */
  stageLabel: string | null;
  /** Guarded last-activity timestamp; null when unknown. */
  lastActivityAt: string | null;
  /** "Running · 5 minutes ago". */
  meta: string;
}

/** Display name, phase, and "status · time" line shared by session rows. */
export function useSessionMeta(
  session: MarsSession,
  stage: MarsConnectStage | null = null,
): SessionMeta {
  const { t } = useTranslation("openhands");
  const phase = getSessionPhase(session.status);
  const phaseLabel = t(PHASE_LABEL_KEY[phase]);
  const activity = useSessionActivityLabel(session);
  return {
    name: getSessionDisplayName(session, t),
    phase,
    phaseLabel,
    stageLabel: stage ? t(STAGE_LABEL_KEY[stage]) : null,
    lastActivityAt: getSessionLastActivity(session),
    meta: [phaseLabel, activity].filter(Boolean).join(" · "),
  };
}
