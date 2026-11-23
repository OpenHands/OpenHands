import type { TFunction } from "i18next";

import type {
  MarsAgentConfig,
  MarsConnection,
  MarsSession,
  MarsSessionPhase,
} from "#/api/mars/mars-tunnel-backend";
import type { MarsConnectStage } from "#/hooks/use-connect-mars-session";
import { I18nKey } from "#/i18n/declaration";

const SHORT_ID_LENGTH = 8;

export const PHASE_DOT_CLASS: Record<MarsSessionPhase, string> = {
  ready:
    "bg-status-success shadow-[0_0_0_3px_color-mix(in_srgb,var(--oh-status-success)_22%,transparent)]",
  paused: "bg-warning",
  starting: "bg-interactive-selected animate-pulse",
  failed: "bg-status-error",
  ended: "bg-text-tertiary",
};

export const PHASE_LABEL_KEY: Record<MarsSessionPhase, I18nKey> = {
  ready: I18nKey.COMMON$RUNNING,
  paused: I18nKey.DO_AGENTS$STATUS_PAUSED,
  starting: I18nKey.COMMON$STARTING,
  failed: I18nKey.DO_AGENTS$STATUS_FAILED,
  ended: I18nKey.DO_AGENTS$STATUS_ENDED,
};

export const STAGE_LABEL_KEY: Record<MarsConnectStage, I18nKey> = {
  waking: I18nKey.DO_AGENTS$STAGE_WAKING,
  tunnel: I18nKey.DO_AGENTS$STAGE_TUNNEL,
  agent: I18nKey.DO_AGENTS$STAGE_AGENT,
};

/** "Team · email" for a signed-in connection, falling back to its label. */
export function connectionLabel(connection: MarsConnection): string {
  return [connection.teamName ?? connection.label, connection.email]
    .filter(Boolean)
    .join(" · ");
}

export function shortSessionId(session: MarsSession): string {
  return session.session_id.slice(0, SHORT_ID_LENGTH);
}

export function agentLabel(config: MarsAgentConfig): string {
  return config.name?.trim() || config.id;
}

export function getSessionDisplayName(
  session: MarsSession,
  t: TFunction<"openhands">,
): string {
  return (
    session.name?.trim() ||
    t(I18nKey.DO_AGENTS$SESSION_NAME_FALLBACK, { id: shortSessionId(session) })
  );
}

/**
 * Just the session's name: sessions launched here already carry their agent's
 * name as a prefix, so repeating the agent only cluttered the backend menu.
 */
export function buildMarsBackendName(
  session: MarsSession,
  t: TFunction<"openhands">,
): string {
  return getSessionDisplayName(session, t);
}

/**
 * Plural key (`_zero` … `_other` entries in translation.json), so it is not
 * an `I18nKey` member; pass `{ count }`.
 */
export const SESSION_COUNT_I18N_KEY = "DO_AGENTS$SESSION_COUNT";
