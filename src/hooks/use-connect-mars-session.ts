import React from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import { withBackendSelectionParams } from "#/api/backend-registry/url-selection";
import {
  getMarsErrorMessage,
  getSessionPhase,
  MarsAttachError,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import { useNavigation } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { useMarsTunnelBackend } from "./use-mars-tunnel-backend";

export type MarsConnectStage = "waking" | "tunnel" | "agent";

export interface MarsConnectingState {
  sessionId: string;
  stage: MarsConnectStage;
}

export interface ConnectMarsSessionParams {
  session: MarsSession;
  /** Display name for the registered backend. */
  name: string;
  configId?: string;
}

const NEW_CHAT_PATH = "/conversations";

const ATTACH_FAILURE_MESSAGE_KEY = {
  "not-openhands": I18nKey.DO_AGENTS$ERROR_NOT_OPENHANDS,
  refused: I18nKey.DO_AGENTS$ERROR_REFUSED,
} as const;

/** User-facing explanation for a failed connect. */
export function getMarsConnectErrorMessage(
  error: unknown,
  t: TFunction<"openhands">,
): string {
  if (error instanceof MarsAttachError) {
    return t(ATTACH_FAILURE_MESSAGE_KEY[error.reason]);
  }
  return (
    getMarsErrorMessage(error) ?? t(I18nKey.BACKEND$DIGITALOCEAN_ATTACH_FAILED)
  );
}

/**
 * Connect to a MARS session and land the user in it: the session's most
 * recent conversation when it has one, otherwise a fresh chat. Tracks which
 * stage the connect is in so callers can show real progress, since waking a
 * paused sandbox can take tens of seconds.
 */
export function useConnectMarsSession() {
  const { t } = useTranslation("openhands");
  const { attach } = useMarsTunnelBackend();
  const { navigate } = useNavigation();
  const [connecting, setConnecting] =
    React.useState<MarsConnectingState | null>(null);

  const connect = React.useCallback(
    async ({ session, name, configId }: ConnectMarsSessionParams) => {
      const sessionId = session.session_id;
      setConnecting({
        sessionId,
        stage:
          getSessionPhase(session.status) === "paused" ? "waking" : "tunnel",
      });
      try {
        const { backend, latestConversationId } = await attach({
          sessionId,
          name,
          configId,
          onTunnelReady: () => setConnecting({ sessionId, stage: "agent" }),
        });
        const path = latestConversationId
          ? `/conversations/${latestConversationId}`
          : NEW_CHAT_PATH;
        // Pin the URL to the new backend so the route resolves against this
        // session even while the registry switch is still propagating.
        navigate(withBackendSelectionParams(path, { backend, orgId: null }));
        return backend;
      } finally {
        setConnecting(null);
      }
    },
    [attach, navigate],
  );

  const describeError = React.useCallback(
    (error: unknown) => getMarsConnectErrorMessage(error, t),
    [t],
  );

  return { connect, connecting, describeError };
}
