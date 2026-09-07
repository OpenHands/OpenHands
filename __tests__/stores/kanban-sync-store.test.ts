import { beforeEach, describe, expect, it } from "vitest";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import { useKanbanSyncStore } from "#/stores/kanban-sync-store";
import { KANBAN_SYNC_STORAGE_KEY } from "#/utils/kanban-sync";

function makeCard(overrides: Partial<KanbanCard> = {}): KanbanCard {
  return {
    id: "card-1",
    column_id: "col-1",
    board_id: "board-1",
    title: "Ship login",
    description: "Local copy",
    priority: "P2",
    status: "todo",
    assignee: null,
    linked_branch: null,
    linked_pr: null,
    estimate_tokens: null,
    estimate_cost: null,
    actual_tokens: null,
    actual_cost: null,
    model_used: null,
    tool_calls: null,
    agent_time: null,
    agent_session_id: null,
    position: 0,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

describe("kanban sync store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useKanbanSyncStore.getState().reset();
  });

  it("maps a source, creates remote swimlanes, and opens reconcile", () => {
    const source = useKanbanSyncStore.getState().mapSource({
      boardId: "board-1",
      provider: "linear",
      label: "Linear",
      master: "remote",
      issues: [
        {
          sourceId: "pending",
          externalId: "LIN-1",
          title: "Ship login",
          description: "Remote copy",
          status: "todo",
          assignee: null,
          url: null,
          groupKey: "auth",
          groupLabel: "Auth",
          updatedAt: "2026-09-02T00:00:00Z",
        },
      ],
      localCards: [makeCard()],
    });

    const state = useKanbanSyncStore.getState();
    expect(source.provider).toBe("linear");
    expect(state.mappingOpen).toBe(false);
    expect(state.reconcileOpen).toBe(true);
    expect(state.lanes.map((lane) => lane.name)).toEqual(["Auth"]);
    expect(state.conflicts.some((item) => item.kind === "extend_local")).toBe(
      true,
    );
    expect(window.localStorage.getItem(KANBAN_SYNC_STORAGE_KEY)).toContain(
      "Linear",
    );
  });

  it("treats the chosen master as the recommended side", () => {
    useKanbanSyncStore.getState().mapSource({
      boardId: "board-1",
      provider: "jira",
      label: "Jira",
      master: "local",
      issues: [
        {
          sourceId: "pending",
          externalId: "JIRA-1",
          title: "Ship login",
          description: "Jira copy",
          status: "todo",
          assignee: null,
          url: null,
          groupKey: null,
          groupLabel: null,
          updatedAt: "2026-09-02T00:00:00Z",
        },
      ],
      localCards: [makeCard()],
    });

    expect(useKanbanSyncStore.getState().conflicts[0]?.recommended).toBe(
      "local",
    );
    expect(useKanbanSyncStore.getState().conflicts[0]?.kind).toBe(
      "extend_remote",
    );
  });

  it("reopens reconcile when a local card conflicts with a mapped source", () => {
    useKanbanSyncStore.getState().mapSource({
      boardId: "board-1",
      provider: "linear",
      label: "Linear",
      master: "remote",
      issues: [
        {
          sourceId: "pending",
          externalId: "LIN-9",
          title: "New remote issue",
          description: null,
          status: "todo",
          assignee: null,
          url: null,
          groupKey: null,
          groupLabel: null,
          updatedAt: "2026-09-02T00:00:00Z",
        },
      ],
      localCards: [],
    });
    useKanbanSyncStore.setState({ reconcileOpen: false, conflicts: [] });

    useKanbanSyncStore
      .getState()
      .detectAfterLocalChange("board-1", [
        makeCard({ title: "Brand new local" }),
      ]);

    expect(useKanbanSyncStore.getState().reconcileOpen).toBe(true);
    expect(
      useKanbanSyncStore
        .getState()
        .conflicts.some(
          (item) =>
            item.kind === "create_remote" || item.kind === "create_local",
        ),
    ).toBe(true);
  });
});
