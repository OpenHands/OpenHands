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

interface ActiveBackendContextValue {
  backends: Backend[];
  active: ResolvedActiveBackend;
  setActive: (backendId: string, orgId?: string | null) => void;
  addBackend: (backend: Omit<Backend, "id">) => Backend;
  updateBackend: (id: string, patch: Partial<Omit<Backend, "id">>) => void;
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
      const prevBackendId = getActiveSelection()?.backendId ?? null;
      const prevOrgId = getActiveSelection()?.orgId ?? null;
      const nextOrgId = orgId ?? null;

      if (backendId === prevBackendId && nextOrgId === prevOrgId) return;

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
    (backend: Omit<Backend, "id">): Backend => {
      const next: Backend = { ...backend, id: generateId() };
      const list = [...getRegisteredBackends(), next];
      setRegisteredBackends(list);
      setActiveSelection({ backendId: next.id });
      retryBootstrapProbe();
      return next;
    },
    [retryBootstrapProbe],
  );

  const updateBackend = React.useCallback(
    (id: string, patch: Partial<Omit<Backend, "id">>) => {
      const prev = getRegisteredBackends().find((b) => b.id === id);
      const activeBeforeUpdate = getActiveSelection()?.backendId ?? null;
      const list = getRegisteredBackends().map((b) =>
        b.id === id ? { ...b, ...patch } : b,
      );
      setRegisteredBackends(list);

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
      if (hostChanged || apiKeyChanged) {
        resetBackendHealth(id);
        if (activeBeforeUpdate === id) {
          retryBootstrapProbe();
        }
      }
    },
    [retryBootstrapProbe],
  );

  const removeBackend = React.useCallback(
    (id: string) => {
      const list = getRegisteredBackends().filter((b) => b.id !== id);
      setRegisteredBackends(list);
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
