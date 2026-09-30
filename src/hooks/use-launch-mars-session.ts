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
  type NewMarsAgentInput,
} from "#/api/mars/mars-tunnel-backend";
import { buildMarsBackendName } from "#/components/features/backends/managed-agents/managed-agents-labels";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import type { MarsAgentGroup } from "#/hooks/query/use-mars-agents";
import { useConnectMarsSession } from "#/hooks/use-connect-mars-session";
import { I18nKey } from "#/i18n/declaration";

/** Creating an agent or session failed, as opposed to connecting afterwards. */
class MarsCreateError extends Error {
  constructor(
    readonly original: unknown,
    readonly fallbackKey: I18nKey,
  ) {
    super(getMarsErrorMessage(original) ?? "");
    this.name = "MarsCreateError";
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
  const [isCreatingAgent, setIsCreatingAgent] = React.useState(false);

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
        throw new MarsCreateError(
          error,
          I18nKey.BACKEND$DIGITALOCEAN_SESSION_CREATE_FAILED,
        );
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

  /**
   * Create an OpenHands agent and wait for the list to include it, so the
   * caller can launch its first session from a row that is already visible.
   */
  const createAgent = React.useCallback(
    async (input: NewMarsAgentInput) => {
      setIsCreatingAgent(true);
      try {
        const config = await getMarsBridge()!.createOpenHandsAgent(input);
        await queryClient.invalidateQueries({
          queryKey: MARS_QUERY_KEYS.allAgents,
        });
        return config;
      } catch (error) {
        throw new MarsCreateError(error, I18nKey.DO_AGENTS$AGENT_CREATE_FAILED);
      } finally {
        setIsCreatingAgent(false);
      }
    },
    [queryClient],
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
      error instanceof MarsCreateError
        ? error.message || t(error.fallbackKey)
        : describeConnectError(error),
    [describeConnectError, t],
  );

  return {
    open,
    launch,
    createAgent,
    openAgent,
    connecting,
    launchingConfigId,
    isCreatingAgent,
    isBusy:
      connecting !== null || launchingConfigId !== null || isCreatingAgent,
    describeError,
  };
}
