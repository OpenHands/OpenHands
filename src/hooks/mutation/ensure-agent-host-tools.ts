import type { QueryClient } from "@tanstack/react-query";

import { getActiveBackend } from "#/api/backend-registry/active-store";
import { SERVER_CHILD_CONVERSATION_TOOL_NAME } from "#/constants/child-conversation";
import {
  buildProfileToolsValue,
  readProfileTools,
  type ProfileToolSpec,
} from "#/constants/profile-tools";
import { toolCatalogQueryOptions } from "#/hooks/query/use-tool-catalog";
import { updateAgentProfile } from "#/hooks/mutation/sync-agent-profiles";

/**
 * Agent-server tools the host attaches instead of the user picking them from
 * the catalog. Canvas launches by agent profile, and a profile launch takes its
 * tool set from the profile, so writing the profile is the only launch-time
 * channel Canvas has for them. The child launcher is registered by the server
 * (`user_selectable: false`, `in_default_set: false`) but stays inert until it
 * is on the agent.
 */
const HOST_TOOL_NAMES: readonly string[] = [
  SERVER_CHILD_CONVERSATION_TOOL_NAME,
];

const toToolsChange = (
  tools: ProfileToolSpec[] | null,
): { tools: ProfileToolSpec[] } | null => (tools ? { tools } : null);

/** Host tools this backend advertises as runnable, in the server's order. */
export function usableHostTools(
  catalog: readonly { name: string; usable: boolean }[],
): string[] {
  const usable = new Set(
    catalog.filter(({ usable: canRun }) => canRun).map(({ name }) => name),
  );
  return HOST_TOOL_NAMES.filter((name) => usable.has(name));
}

/**
 * The profile's `tools` with the host tools added, or `null` when it already
 * has them all. A profile on the standard set stores `null`, which cannot
 * express "the standard set plus one tool", so the standard set is pinned
 * alongside the host tool — the same trade the profile editor makes when a user
 * turns a tool on.
 */
export function withHostTools(
  tools: unknown,
  hostTools: readonly string[],
  standardTools: readonly string[],
): ProfileToolSpec[] | null {
  const stored = readProfileTools(tools);
  const missing = hostTools.filter((name) => !stored.selected.includes(name));
  if (missing.length === 0) return null;
  const base = stored.mode === "standard" ? standardTools : stored.selected;
  return buildProfileToolsValue({
    mode: "custom",
    selected: [...base, ...missing],
    params: stored.params,
  });
}

async function attachHostTools(
  queryClient: QueryClient,
  profileName: string,
): Promise<boolean> {
  const { backend } = getActiveBackend();
  const catalog = await queryClient
    .ensureQueryData(toolCatalogQueryOptions(backend))
    .catch(() => undefined);
  if (!catalog) return false;
  const hostTools = usableHostTools(catalog);
  if (hostTools.length === 0) return false;
  const standardTools = catalog
    .filter(({ in_default_set: inDefault, usable }) => inDefault && usable)
    .map(({ name }) => name);
  await updateAgentProfile(
    queryClient,
    profileName,
    (stored) =>
      stored.agent_kind === "openhands"
        ? toToolsChange(withHostTools(stored.tools, hostTools, standardTools))
        : null,
    {
      // Attaching is best effort: a launch must not fail over it.
      onError: (error: unknown) => {
        console.warn(
          "Could not attach the host tools to the agent profile:",
          error,
        );
      },
    },
  );
  return true;
}

/** Profiles already attempted, per query client, as `backend:profile`. */
const attemptedProfiles = new WeakMap<QueryClient, Set<string>>();

/**
 * Resolve once the profile carries the host tools the active backend offers.
 * Runs at most once per profile and session; nothing else writes these names.
 */
export function ensureAgentHostTools(
  queryClient: QueryClient,
  profileName: string,
): Promise<boolean> {
  const { backend } = getActiveBackend();
  const key = `${backend.id}:${profileName}`;
  let attempted = attemptedProfiles.get(queryClient);
  if (!attempted) {
    attempted = new Set();
    attemptedProfiles.set(queryClient, attempted);
  }
  // Claimed before the request: concurrent launches share one attempt.
  if (attempted.has(key)) return Promise.resolve(true);
  attempted.add(key);
  return attachHostTools(queryClient, profileName);
}
