import { describe, it, expect, beforeEach, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

import { useAutoRefreshFilesOnEdit } from "#/hooks/use-auto-refresh-files-on-edit";
import { useEventStore } from "#/stores/use-event-store";
import type { OHEvent } from "#/stores/use-event-store";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";

function makeWrapper(client: QueryClient) {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

function makeObservationEvent(
  id: string,
  kind: string,
  command: string,
): OHEvent {
  return {
    id,
    timestamp: new Date(Date.now() + Number(id.replace(/\D/g, "")) * 1000)
      .toISOString(),
    source: "environment",
    tool_name: "str_replace_based_edit_tool",
    tool_call_id: `tc-${id}`,
    action_id: `act-${id}`,
    observation: {
      kind,
      command,
      path: "/workspace/project/foo.txt",
      old_content: null,
      new_content: "hello",
      output: "ok",
    },
  } as unknown as OHEvent;
}

describe("useAutoRefreshFilesOnEdit", () => {
  beforeEach(() => {
    act(() => {
      useEventStore.getState().clearEvents();
      // Reset the workspace mutation counter so per-test counter assertions
      // don't see ticks bled over from earlier tests.
      useWorkspaceMutationCounter.setState({ count: 0 });
    });
  });

  it("invalidates workspace queries when a mutating file editor observation arrives", () => {
    const client = new QueryClient();
    const spy = vi.spyOn(client, "invalidateQueries");

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    expect(spy).not.toHaveBeenCalled();

    act(() => {
      useEventStore
        .getState()
        .addEvent(
          makeObservationEvent("1", "FileEditorObservation", "str_replace"),
        );
    });

    const invalidatedKeys = spy.mock.calls.map(
      (call) => (call[0] as { queryKey: unknown[] }).queryKey[0],
    );
    expect(invalidatedKeys).toContain("workspace-files");
    expect(invalidatedKeys).toContain("workspace-file-content");
    expect(invalidatedKeys).toContain("file_changes");
    expect(invalidatedKeys).toContain("file_diff");
  });

  it.each(["ExecuteBashObservation", "TerminalObservation"])(
    "refreshes only the git diff queries when a %s arrives",
    (kind) => {
      // Arrange
      const client = new QueryClient();
      const spy = vi.spyOn(client, "invalidateQueries");
      renderHook(() => useAutoRefreshFilesOnEdit(), {
        wrapper: makeWrapper(client),
      });

      // Act — bash observations are how `git commit` / shell file edits
      // reach the event stream.
      act(() => {
        useEventStore
          .getState()
          .addEvent(makeObservationEvent("1", kind, "git commit -m 'done'"));
      });

      // Assert — the diff and commit-list queries refresh, and nothing
      // else does (workspace file queries on every shell command would
      // churn the Files tab; per-commit queries are immutable).
      const invalidatedKeys = spy.mock.calls.map(
        (call) => (call[0] as { queryKey: unknown[] }).queryKey[0],
      );
      expect(invalidatedKeys).toEqual([
        "file_changes",
        "file_diff",
        "git_commits",
      ]);
    },
  );

  it("does not apply the file-editor read-only filter to bash commands", () => {
    // Arrange — `READ_ONLY_COMMANDS` matches editor sub-commands; for bash
    // observations `command` is a whole shell command line and must not be
    // filtered even if it collides with an editor sub-command name.
    const client = new QueryClient();
    const spy = vi.spyOn(client, "invalidateQueries");
    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    // Act
    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("1", "ExecuteBashObservation", "view"));
    });

    // Assert
    expect(spy).toHaveBeenCalled();
  });

  it("ignores read-only `view` observations", () => {
    const client = new QueryClient();
    const spy = vi.spyOn(client, "invalidateQueries");

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("1", "FileEditorObservation", "view"));
    });

    expect(spy).not.toHaveBeenCalled();
  });

  it("ignores observation kinds that are neither file-editor nor bash", () => {
    const client = new QueryClient();
    const spy = vi.spyOn(client, "invalidateQueries");

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("1", "BrowserObservation", "navigate"));
    });

    expect(spy).not.toHaveBeenCalled();
  });

  it("bumps the workspace mutation counter on each mutating observation so iframes / images cache-bust", () => {
    const client = new QueryClient();

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    expect(useWorkspaceMutationCounter.getState().count).toBe(0);

    act(() => {
      useEventStore
        .getState()
        .addEvent(
          makeObservationEvent("1", "FileEditorObservation", "str_replace"),
        );
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(1);

    act(() => {
      useEventStore
        .getState()
        .addEvent(
          makeObservationEvent(
            "2",
            "StrReplaceEditorObservation",
            "create",
          ),
        );
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(2);
  });

  it("does NOT bump the workspace mutation counter for read-only / non-file observations", () => {
    const client = new QueryClient();

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("1", "FileEditorObservation", "view"));
      useEventStore
        .getState()
        .addEvent(
          makeObservationEvent("2", "ExecuteBashObservation", "ls"),
        );
    });

    expect(useWorkspaceMutationCounter.getState().count).toBe(0);
  });

  it("still reacts to mutations that arrive out-of-order (older timestamp inserted between newer events)", () => {
    // Regression test for a bug where the hook used `events.slice(processedCount)`
    // to find new events. The event store re-sorts by timestamp on insert,
    // so a late-arriving older event lands *between* two newer ones and
    // the tail slice would miss it.
    const client = new QueryClient();
    const spy = vi.spyOn(client, "invalidateQueries");

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    // First, push two newer events. The id-numbers drive the timestamp,
    // so id "10" is later than id "5". Both land in the same effect run
    // (we coalesce — one bump per batch, not per event), so count goes
    // from 0 → 1.
    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("10", "FileEditorObservation", "create"));
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("20", "FileEditorObservation", "create"));
    });
    const callsAfterInitial = spy.mock.calls.length;
    expect(callsAfterInitial).toBeGreaterThan(0);
    const countAfterInitial = useWorkspaceMutationCounter.getState().count;
    expect(countAfterInitial).toBe(1);

    // Now insert an OLDER event (id "5" → earliest timestamp). The store
    // re-sorts so the events array becomes [e5, e10, e20]. The previous
    // "slice from index 2" approach would return [e20] only and miss e5
    // entirely — no invalidation, no cache-bust, stale iframe.
    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("5", "FileEditorObservation", "create"));
    });

    // We should have invalidated again and bumped the counter exactly once
    // more for the late-arriving mutation (count: 1 → 2).
    expect(spy.mock.calls.length).toBeGreaterThan(callsAfterInitial);
    expect(useWorkspaceMutationCounter.getState().count).toBe(
      countAfterInitial + 1,
    );
  });

  it("processes each id-less event distinctly (does NOT collapse them via an `undefined` Set key)", () => {
    // The event store explicitly allows events without ids
    // (`getEventId` returns undefined for them). If the hook keyed dedup
    // on `event.id` naively, a single `undefined` entry in the Set would
    // swallow every subsequent id-less event — silently dropping real
    // mutations on the floor.
    //
    // Verifies via three SEPARATE act() calls (one per event) so each
    // store mutation gets its own effect-flush. The counter bumps once
    // per flush that found at least one new mutation; three flushes →
    // counter ends at 3. Putting all three addEvent calls inside a
    // single act() would batch them into one flush (counter=1) and
    // verify nothing useful.
    const client = new QueryClient();

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    // Three distinct id-less FileEditorObservation events (different
    // timestamps so the store treats them as ordered, not duplicates).
    const idlessEvent = (i: number): OHEvent =>
      ({
        // no `id` field at all → getEventId returns undefined
        timestamp: new Date(2026, 0, 1, 0, 0, i).toISOString(),
        source: "environment",
        tool_name: "str_replace_based_edit_tool",
        tool_call_id: `tc-idless-${i}`,
        action_id: `act-idless-${i}`,
        observation: {
          kind: "FileEditorObservation",
          command: "create",
          path: `/workspace/project/foo${i}.txt`,
          old_content: null,
          new_content: "hello",
          output: "ok",
        },
      }) as unknown as OHEvent;

    act(() => {
      useEventStore.getState().addEvent(idlessEvent(1));
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(1);

    act(() => {
      useEventStore.getState().addEvent(idlessEvent(2));
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(2);

    act(() => {
      useEventStore.getState().addEvent(idlessEvent(3));
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(3);
  });

  it("does NOT re-bump on subsequent renders for the same id-less event", () => {
    // Companion to the previous test, targeting the *other* half of the
    // id-less dedup contract: each id-less event must be processed
    // exactly ONCE across the lifetime of the hook. Without
    // reference-based dedup (`processedEventsRef` WeakSet) the events
    // array — which is rebuilt on every store mutation but keeps stable
    // element references — would cause the same id-less event to
    // re-trigger the bump on every subsequent re-render, spamming
    // cache invalidations.
    const client = new QueryClient();

    const { rerender } = renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    const idlessEvent: OHEvent = {
      timestamp: new Date(2026, 0, 1, 0, 0, 0).toISOString(),
      source: "environment",
      tool_name: "str_replace_based_edit_tool",
      tool_call_id: "tc-idless-stable",
      action_id: "act-idless-stable",
      observation: {
        kind: "FileEditorObservation",
        command: "create",
        path: "/workspace/project/foo.txt",
        old_content: null,
        new_content: "hello",
        output: "ok",
      },
    } as unknown as OHEvent;

    act(() => {
      useEventStore.getState().addEvent(idlessEvent);
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(1);

    // Force several extra re-renders without adding new events. The
    // id-less event still sits in the events array on every re-render,
    // but the WeakSet dedup must prevent it from being re-processed.
    rerender();
    rerender();
    rerender();
    expect(useWorkspaceMutationCounter.getState().count).toBe(1);
  });

  it("dedupes numeric event ids the same way as string ids", () => {
    // The formal EventID type is `string`, but the event store carries
    // `Set<string | number>` defensively (use-event-store.ts:52) and
    // `getEventId` returns `string | number | undefined`. The hook's
    // processed-ids set is widened to match — a stray numeric id (legacy
    // payload, hand-crafted test event, …) must still dedup correctly.
    const client = new QueryClient();

    renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    const numericEvent: OHEvent = {
      id: 42 as unknown as string, // intentionally numeric at runtime
      timestamp: new Date(2026, 0, 1, 0, 0, 1).toISOString(),
      source: "environment",
      tool_name: "str_replace_based_edit_tool",
      tool_call_id: "tc-num",
      action_id: "act-num",
      observation: {
        kind: "FileEditorObservation",
        command: "create",
        path: "/workspace/project/foo.txt",
        old_content: null,
        new_content: "hello",
        output: "ok",
      },
    } as unknown as OHEvent;

    act(() => {
      useEventStore.getState().addEvent(numericEvent);
    });
    const afterFirst = useWorkspaceMutationCounter.getState().count;
    expect(afterFirst).toBe(1);

    // Re-adding the same numeric-id event must be a no-op for the
    // counter (store dedups on id; hook must too).
    act(() => {
      useEventStore.getState().addEvent({ ...numericEvent });
    });
    expect(useWorkspaceMutationCounter.getState().count).toBe(afterFirst);
  });

  it("only invalidates once per new event batch", () => {
    const client = new QueryClient();
    const spy = vi.spyOn(client, "invalidateQueries");

    const { rerender } = renderHook(() => useAutoRefreshFilesOnEdit(), {
      wrapper: makeWrapper(client),
    });

    act(() => {
      useEventStore
        .getState()
        .addEvent(makeObservationEvent("1", "FileEditorObservation", "create"));
    });

    const callsAfterFirst = spy.mock.calls.length;
    expect(callsAfterFirst).toBeGreaterThan(0);

    // Re-render without adding new events — should not re-invalidate.
    rerender();
    expect(spy.mock.calls.length).toBe(callsAfterFirst);
  });
});
