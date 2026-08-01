import React from "react";
import { clearCachedAgentServerInfo } from "#/api/agent-server-compatibility";
import {
  getActiveSelection,
  getRegisteredBackends,
  getSnapshot,
  NO_BACKEND,
  setActiveSelection,
  setRegisteredBackends,
  subscribeActiveBackend,
} from "#/api/backend-registry/active-store";
import { makeDefaultLocalBackend } from "#/api/backend-registry/default-backend";
import {
  dropBackendHealth,
  resetBackendHealth,
} from "#/api/backend-registry/health-store";
import {
  type Backend,
  type BackendSelection,
  type ResolvedActiveBackend,
} from "#/api/backend-registry/types";
import { QUERY_KEYS } from "#/hooks/query/query-keys";
import { queryClient } from "#/query-client-config";
import {
  setTelemetryCloudContext,
  setTelemetryIdentity,
} from "#/services/telemetry";

type BackendInput = Omit<Backend, "id" | "connectionRevision">;

interface ActiveBackendContextValue {
  backends: Backend[];
  active: ResolvedActiveBackend;
  setActive: (backendId: string, orgId?: string | null) => void;
  addBackend: (backend: BackendInput) => Backend;
  updateBackend: (id: string, patch: Partial<BackendInput>) => void;
  removeBackend: (id: string) => void;
}

const ActiveBackendContext =
  React.createContext<ActiveBackendContextValue | null>(null);

function generateId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `backend-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function applyTelemetrySelectionBoundary(
  previous: ResolvedActiveBackend,
  next: ResolvedActiveBackend,
): void {
  const previousWasCloud = previous.backend.kind === "cloud";
  const nextIsCloud = next.backend.kind === "cloud";
  if (!previousWasCloud && !nextIsCloud) return;

  setTelemetryCloudContext(null);

  if (
    nextIsCloud &&
    (!previousWasCloud || previous.backend.id !== next.backend.id)
  ) {
    void setTelemetryIdentity(null);
  }
}

export function ActiveBackendProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const snapshot = React.useSyncExternalStore(
    subscribeActiveBackend,
    getSnapshot,
    getSnapshot,
  );

  const retryBootstrapProbe = React.useCallback(() => {
    clearCachedAgentServerInfo();
    void queryClient.invalidateQueries({
      queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    });
  }, []);

  const setActive = React.useCallback(
    (backendId: string, orgId?: string | null) => {
      const currentSnapshot = getSnapshot();
      const prevBackendId = getActiveSelection()?.backendId ?? null;
      const prevOrgId = getActiveSelection()?.orgId ?? null;
      const nextOrgId = orgId ?? null;

      if (backendId === prevBackendId && nextOrgId === prevOrgId) return;

      const registeredBackends = getRegisteredBackends();
      const selectedBackend = registeredBackends.find(
        (b) => b.id === backendId,
      );
      const nextActive: ResolvedActiveBackend = {
        backend: selectedBackend ?? registeredBackends[0] ?? NO_BACKEND,
        orgId: selectedBackend ? nextOrgId : null,
      };
      applyTelemetrySelectionBoundary(currentSnapshot.active, nextActive);

      const next: BackendSelection = { backendId, orgId: nextOrgId };
      setActiveSelection(next);
      retryBootstrapProbe();

      // No blanket `invalidateQueries()` here. Long-lived queries
      // (`useSettings`, `usePaginatedConversations`,
      // `useGitRepositories`, `useAppInstallations`,
      // `useCloudCurrentUserId`, `useGitUser`, …) include the active
      // backend's `id` and `orgId` in their query keys, so React Query
      // treats a backend/org switch as a brand-new query and fetches
      // automatically — once, with no duplicate waves.
    },
    [retryBootstrapProbe],
  );

  // @spec BM-001 — Auto-switch to newly connected backend
  const addBackend = React.useCallback(
    (backend: BackendInput): Backend => {
      const previousActive = getSnapshot().active;
      const next: Backend = { ...backend, id: generateId() };
      const list = [...getRegisteredBackends(), next];
      applyTelemetrySelectionBoundary(previousActive, {
        backend: next,
        orgId: null,
      });
      setRegisteredBackends(list);
      setActiveSelection({ backendId: next.id });
      retryBootstrapProbe();
      return next;
    },
    [retryBootstrapProbe],
  );

  const updateBackend = React.useCallback(
    (id: string, patch: Partial<BackendInput>) => {
      const prev = getRegisteredBackends().find((b) => b.id === id);
      const activeBeforeUpdate = getSnapshot().active.backend.id;
      // Re-arm health polling when the user edits the fields that
      // actually drive the probe. Cosmetic edits (name) shouldn't
      // re-enable a backend that was disabled for being unreachable.
      const hostChanged =
        patch.host !== undefined &&
        prev !== undefined &&
        patch.host !== prev.host;
      const apiKeyChanged =
        patch.apiKey !== undefined &&
        prev !== undefined &&
        patch.apiKey !== prev.apiKey;
      const list = getRegisteredBackends().map((b) =>
        b.id === id
          ? {
              ...b,
              ...patch,
              connectionRevision:
                hostChanged || apiKeyChanged
                  ? (b.connectionRevision ?? 0) + 1
                  : b.connectionRevision,
            }
          : b,
      );
      setRegisteredBackends(list);

      if (hostChanged || apiKeyChanged) {
        resetBackendHealth(id);
        if (activeBeforeUpdate === id) {
          if (prev?.kind === "cloud") {
            setTelemetryCloudContext(null);
            void setTelemetryIdentity(null);
          }
          retryBootstrapProbe();
        }
      }
    },
    [retryBootstrapProbe],
  );

  const removeBackend = React.useCallback(
    (id: string) => {
      const removed = getRegisteredBackends().find((b) => b.id === id);
      const wasActiveCloud =
        removed?.kind === "cloud" &&
        getSnapshot().active.backend.id === removed.id;
      const list = getRegisteredBackends().filter((b) => b.id !== id);
      setRegisteredBackends(list);
      if (wasActiveCloud) {
        setTelemetryCloudContext(null);
        void setTelemetryIdentity(null);
      }
      dropBackendHealth(id);
      retryBootstrapProbe();
      // If the active selection pointed at this backend, the active
      // store falls back to the first remaining local backend (or the
      // env-derived default if no locals exist); consumer hooks re-key
      // by the new active backend identity and refetch automatically.
    },
    [retryBootstrapProbe],
  );

  const value = React.useMemo<ActiveBackendContextValue>(
    () => ({
      backends: snapshot.backends,
      active: snapshot.active,
      setActive,
      addBackend,
      updateBackend,
      removeBackend,
    }),
    [snapshot, setActive, addBackend, updateBackend, removeBackend],
  );

  return (
    <ActiveBackendContext.Provider value={value}>
      {children}
    </ActiveBackendContext.Provider>
  );
}

export function useActiveBackendContext(): ActiveBackendContextValue {
  const ctx = React.useContext(ActiveBackendContext);
  if (!ctx) {
    throw new Error(
      "useActiveBackendContext must be used inside <ActiveBackendProvider>",
    );
  }
  return ctx;
}

/**
 * Read the resolved active backend.
 *
 * Falls back to a synthesized env-derived local backend when called
 * outside an `<ActiveBackendProvider>` (e.g. from a unit test that
 * mounts a narrow component without the full provider stack). That
 * synthesized backend is identical to the seed used on first install.
 *
 * Components that need to mutate state (`setActive`, `addBackend`,
 * etc.) must use `useActiveBackendContext()` directly — that throws if
 * the provider is missing, since mutation requires the live store.
 */
export function useActiveBackend(): ResolvedActiveBackend {
  const ctx = React.useContext(ActiveBackendContext);
  if (ctx) return ctx.active;
  return { backend: makeDefaultLocalBackend() ?? NO_BACKEND, orgId: null };
}
