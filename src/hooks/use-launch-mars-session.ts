import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  buildNewSessionName,
  getMarsBridge,
  getMarsErrorMessage,
  isConnectableSession,
  type MarsAgentConfig,
  type MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import { buildMarsBackendName } from "#/components/features/backends/managed-agents/managed-agents-labels";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import type { MarsAgentGroup } from "#/hooks/query/use-mars-agents";
import { useConnectMarsSession } from "#/hooks/use-connect-mars-session";
import { I18nKey } from "#/i18n/declaration";

/** Creating the session failed, as opposed to connecting to it afterwards. */
class MarsSessionCreateError extends Error {
  constructor(readonly original: unknown) {
    super(getMarsErrorMessage(original) ?? "");
    this.name = "MarsSessionCreateError";
  }
}

/**
 * Open, launch, and resume Managed Agents sessions, landing the user in the
 * session's latest conversation each time. Every method rejects on failure;
 * pass the error to `describeError` for copy.
 */
export function useLaunchMarsSession() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const {
    connect,
    connecting,
    describeError: describeConnectError,
  } = useConnectMarsSession();
  const [launchingConfigId, setLaunchingConfigId] = React.useState<
    string | null
  >(null);

  const open = React.useCallback(
    (session: MarsSession, config?: MarsAgentConfig) =>
      connect({
        session,
        name: buildMarsBackendName(session, config, t),
        configId: config?.id ?? session.config_id ?? undefined,
      }),
    [connect, t],
  );

  const launch = React.useCallback(
    async (config: MarsAgentConfig) => {
      setLaunchingConfigId(config.id);
      let session: MarsSession;
      try {
        session = await getMarsBridge()!.createSession(
          config.id,
          buildNewSessionName(config.name),
        );
      } catch (error) {
        throw new MarsSessionCreateError(error);
      } finally {
        setLaunchingConfigId(null);
      }
      void queryClient.invalidateQueries({
        queryKey: MARS_QUERY_KEYS.allAgents,
      });
      return open(session, config);
    },
    [open, queryClient],
  );

  /** Resume the agent's most recent live session, or start its first. */
  const openAgent = React.useCallback(
    ({ config, sessions }: MarsAgentGroup) => {
      const latest = sessions.find(isConnectableSession);
      return latest ? open(latest, config) : launch(config);
    },
    [launch, open],
  );

  const describeError = React.useCallback(
    (error: unknown) =>
      error instanceof MarsSessionCreateError
        ? error.message || t(I18nKey.BACKEND$DIGITALOCEAN_SESSION_CREATE_FAILED)
        : describeConnectError(error),
    [describeConnectError, t],
  );

  return {
    open,
    launch,
    openAgent,
    connecting,
    launchingConfigId,
    isBusy: connecting !== null || launchingConfigId !== null,
    describeError,
  };
}
