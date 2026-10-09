import type { TFunction } from "i18next";

import type {
  MarsAgentConfig,
  MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import { I18nKey } from "#/i18n/declaration";

const SHORT_ID_LENGTH = 8;

/** Separates agent and session in a registered backend's display name. */
const BACKEND_NAME_SEPARATOR = " · ";

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

/** "<agent> · <session>" so the backend selector names both. */
export function buildMarsBackendName(
  session: MarsSession,
  config: MarsAgentConfig | undefined,
  t: TFunction<"openhands">,
): string {
  const sessionName = getSessionDisplayName(session, t);
  return config
    ? `${agentLabel(config)}${BACKEND_NAME_SEPARATOR}${sessionName}`
    : sessionName;
}
