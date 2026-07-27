import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";

// `kind` values we treat as a file-mutation observation.
const FILE_EDIT_OBSERVATION_KINDS = new Set([
  "FileEditorObservation",
  "StrReplaceEditorObservation",
  "PlanningFileEditorObservation",
]);

// Commands on the str-replace-editor family that don't change anything on
// disk. We don't want to invalidate caches for those.
const READ_ONLY_COMMANDS = new Set(["view"]);

// Bash observations are how `git commit` / `git push` (and any file edit
// done through the shell) arrive, so they must refresh the Diff view too.
// Legacy agent-servers emit `ExecuteBashObservation`; the current SDK
// emits `TerminalObservation`.
const BASH_OBSERVATION_KINDS = new Set([
  "ExecuteBashObservation",
  "TerminalObservation",
]);

function isFileMutationObservation(event: OHEvent): boolean {
  // ObservationEvents have `source: "environment"` and an `observation`
  // field — narrow to that shape without pulling in the whole event union.
  const obs = (event as { observation?: { kind?: string; command?: string } })
    .observation;
  if (!obs || typeof obs.kind !== "string") return false;
  if (!FILE_EDIT_OBSERVATION_KINDS.has(obs.kind)) return false;
  if (obs.command && READ_ONLY_COMMANDS.has(obs.command)) return false;
  return true;
}

// Note: `READ_ONLY_COMMANDS` intentionally doesn't apply here — for bash
// observations `command` is the whole shell command line, not a
// file-editor sub-command.
function isBashObservation(event: OHEvent): boolean {
  const obs = (event as { observation?: { kind?: string } }).observation;
  return (
    !!obs &&
    typeof obs.kind === "string" &&
    BASH_OBSERVATION_KINDS.has(obs.kind)
  );
}

/**
 * Watches the conversation event stream and invalidates the workspace file
 * queries whenever the agent commits a file-editor mutation (create / edit /
 * insert / undo_edit). This keeps the Files tab's list, content view and
 * diff view in sync with what the agent has actually written to disk,
 * without requiring the user to click refresh manually.
 *
 * Bash observations also refresh the git-diff queries (`file_changes` /
 * `file_diff`) — a `git commit` or `git push` changes what the Diff view
 * should display, and shell commands can edit files too. They deliberately
 * do NOT touch the workspace file queries or the workspace mutation
 * counter: bumping the counter reloads canvas iframes, and doing that for
 * every shell command the agent runs would cause constant flicker.
 * Invalidation only refetches actively-mounted queries, so the cost is
 * limited to when the Files tab is open.
 *
 * Mount this hook inside any component that should drive auto-refresh —
 * the Files tab is the obvious caller. Multiple mounts are safe because
 * React Query coalesces overlapping invalidations.
 */
export function useAutoRefreshFilesOnEdit(): void {
  const queryClient = useQueryClient();
  const events = useEventStore((state) => state.events);
  const bumpWorkspaceMutationCounter = useWorkspaceMutationCounter(
    (state) => state.bump,
  );

  // Track which events we've already reacted to. Two parallel stores:
  //
  // 1. `processedIdsRef` — keys events that *have* an id. The event store
  //    re-sorts on insert when out-of-order events arrive (an older event
  //    can land *between* two newer ones already in the array), so we
  //    cannot use a `slice(processedCount)` trick — it would miss a
  //    late-arriving older event because the array length grew but the
  //    tail we just diffed didn't contain it. Using a Set of ids is O(n)
  //    per render in the worst case but small in practice and immune to
  //    reordering. Type matches the event store's own dedup set
  //    (`Set<string | number>` in `use-event-store.ts`) so a stray
  //    numeric id (legacy server payload, hand-crafted test event, …)
  //    can't sneak past.
  //
  // 2. `processedEventsRef` — keys events that have NO id. We cannot put
  //    a literal `undefined` into the id Set: that would make the second,
  //    third, … id-less arrival collide on the same `undefined` key and
  //    silently skip them. We also cannot just skip dedup for id-less
  //    events: the events array is rebuilt on every store mutation but
  //    its element references are stable, so the same id-less event
  //    appears in the array forever and would re-bump the workspace
  //    mutation counter on every subsequent re-render. Keying by object
  //    reference (via a WeakSet) gives us "process each id-less event
  //    exactly once" — durable, free, no held-onto memory after the
  //    store clears.
  const processedIdsRef = useRef<Set<string | number>>(new Set());
  const processedEventsRef = useRef<WeakSet<OHEvent>>(new WeakSet());

  useEffect(() => {
    let hasNewFileEdits = false;
    let hasNewBashCommands = false;
    for (const event of events) {
      const id: string | number | undefined =
        "id" in event ? event.id : undefined;
      const alreadyProcessed =
        id !== undefined
          ? processedIdsRef.current.has(id)
          : processedEventsRef.current.has(event);
      // Inverted predicate so we avoid `continue` (banned by repo lint).
      if (!alreadyProcessed) {
        if (id !== undefined) {
          processedIdsRef.current.add(id);
        } else {
          processedEventsRef.current.add(event);
        }
        if (isFileMutationObservation(event)) hasNewFileEdits = true;
        else if (isBashObservation(event)) hasNewBashCommands = true;
      }
    }

    if (!hasNewFileEdits && !hasNewBashCommands) return;

    // Editor and bash mutations both change what the git Diff view shows —
    // including the per-file diff content of an already-expanded file.
    // The commits list refreshes too (a `git commit` arrives as a bash
    // observation); per-commit queries (`commit_changes` /
    // `commit_file_diff`) are sha-addressed and immutable, so they are
    // deliberately NOT invalidated.
    queryClient.invalidateQueries({ queryKey: ["file_changes"] });
    queryClient.invalidateQueries({ queryKey: ["file_diff"] });
    queryClient.invalidateQueries({ queryKey: ["git_commits"] });

    if (hasNewFileEdits) {
      queryClient.invalidateQueries({ queryKey: ["workspace-files"] });
      queryClient.invalidateQueries({ queryKey: ["workspace-file-content"] });
      // Force iframes / <img> tags pointing at the static workspace
      // fileserver to re-fetch. Without this they happily keep showing the
      // stale (browser-cached) bytes even after the agent has rewritten the
      // file on disk — e.g. tweaking style.css would silently have no
      // visible effect on the rendered index.html until the user reloaded
      // the whole canvas.
      bumpWorkspaceMutationCounter();
    }
  }, [events, queryClient, bumpWorkspaceMutationCounter]);
}
