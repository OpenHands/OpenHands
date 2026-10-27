import React from "react";
import { useQueryClient } from "@tanstack/react-query";

import { resetBackendHealth } from "#/api/backend-registry/health-store";
import { withBackendSelectionParams } from "#/api/backend-registry/url-selection";
import type { Backend } from "#/api/backend-registry/types";
import {
  buildMaritimeBackendInput,
  fetchLatestMaritimeConversationId,
  NEW_CHAT_PATH,
  requireMaritimeBridge,
  waitForMaritimeAgentServer,
  type MaritimeAgent,
  type MaritimeConnection,
  type NewMaritimeAgentInput,
} from "#/api/maritime/maritime-bridge";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { MARITIME_QUERY_KEYS } from "#/hooks/query/query-keys";

/**
 * What is in flight, so rows can show progress. `waking` covers Maritime
 * starting the VM; `agent` covers waiting for the agent-server inside it.
 */
export type MaritimeBusyState =
  | { kind: "creating"; stage: "machine" | "agent" }
  | {
      kind: "opening" | "resetting";
      agentId: string;
      stage: "machine" | "agent";
    }
  | { kind: "sleeping" | "deleting"; agentId: string };

/**
 * Everything the Maritime screen does to agents. Opening (or creating) an
 * agent registers its agent-server as an ordinary local backend, re-pointing
 * the existing record for that agent when there is one, makes it active and
 * lands the user in its latest conversation.
 */
export function useMaritimeAgentActions({
  onOpened,
}: { onOpened?: (backend: Backend) => void } = {}) {
  const queryClient = useQueryClient();
  const { backends, addBackend, updateBackend, removeBackend, setActive } =
    useActiveBackendContext();
  const { navigate } = useNavigation();
  const [busy, setBusy] = React.useState<MaritimeBusyState | null>(null);

  const invalidate = React.useCallback(
    () => queryClient.invalidateQueries({ queryKey: MARITIME_QUERY_KEYS.all }),
    [queryClient],
  );

  const register = React.useCallback(
    async (connection: MaritimeConnection): Promise<Backend> => {
      await waitForMaritimeAgentServer(connection);
      // Read before the registry switch, while nothing else points here.
      const latestConversationId =
        await fetchLatestMaritimeConversationId(connection);

      const input = buildMaritimeBackendInput(connection);
      const existing = backends.find(
        (b) => b.maritimeAgentId === connection.agentId,
      );
      let backend: Backend;
      if (existing) {
        updateBackend(existing.id, input);
        resetBackendHealth(existing.id);
        setActive(existing.id);
        backend = { ...existing, ...input };
      } else {
        backend = addBackend(input);
      }
      const path = latestConversationId
        ? `/conversations/${latestConversationId}`
        : NEW_CHAT_PATH;
      navigate(withBackendSelectionParams(path, { backend, orgId: null }));
      onOpened?.(backend);
      return backend;
    },
    [addBackend, backends, navigate, onOpened, setActive, updateBackend],
  );

  /** Runs one action, tracking it in `busy`, and refreshes the list after. */
  const run = React.useCallback(
    async <T>(state: MaritimeBusyState, action: () => Promise<T>) => {
      setBusy(state);
      try {
        return await action();
      } finally {
        setBusy(null);
        void invalidate();
      }
    },
    [invalidate],
  );

  const create = React.useCallback(
    (input: NewMaritimeAgentInput) =>
      run({ kind: "creating", stage: "machine" }, async () => {
        const connection = await requireMaritimeBridge().createAgent(input);
        setBusy({ kind: "creating", stage: "agent" });
        return register(connection);
      }),
    [register, run],
  );

  const open = React.useCallback(
    (agent: MaritimeAgent) =>
      run(
        { kind: "opening", agentId: agent.id, stage: "machine" },
        async () => {
          const connection = await requireMaritimeBridge().openAgent(agent.id);
          setBusy({ kind: "opening", agentId: agent.id, stage: "agent" });
          return register(connection);
        },
      ),
    [register, run],
  );

  const resetAccessKey = React.useCallback(
    (agent: MaritimeAgent) =>
      run(
        { kind: "resetting", agentId: agent.id, stage: "machine" },
        async () => {
          const connection = await requireMaritimeBridge().resetAccessKey(
            agent.id,
          );
          setBusy({ kind: "resetting", agentId: agent.id, stage: "agent" });
          return register(connection);
        },
      ),
    [register, run],
  );

  const sleep = React.useCallback(
    (agent: MaritimeAgent) =>
      run({ kind: "sleeping", agentId: agent.id }, () =>
        requireMaritimeBridge().sleepAgent(agent.id),
      ),
    [run],
  );

  /**
   * Delete on Maritime first, then forget the backend: Maritime can refuse,
   * and then the agent still exists and its backend must keep working.
   */
  const remove = React.useCallback(
    (agent: MaritimeAgent) =>
      run({ kind: "deleting", agentId: agent.id }, async () => {
        await requireMaritimeBridge().deleteAgent(agent.id);
        for (const backend of backends) {
          if (backend.maritimeAgentId === agent.id) removeBackend(backend.id);
        }
      }),
    [backends, removeBackend, run],
  );

  return {
    busy,
    isBusy: busy !== null,
    create,
    open,
    resetAccessKey,
    sleep,
    remove,
  };
}
