import { describe, expect, it } from "vitest";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import {
  deriveCodeIssues,
  fingerprintsEqual,
  issueFingerprint,
  parseRemoteIssuesJson,
  reconcileIssues,
  type RemoteIssue,
} from "#/utils/kanban-sync";

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

function makeRemote(overrides: Partial<RemoteIssue> = {}): RemoteIssue {
  return {
    sourceId: "linear-1",
    externalId: "LIN-1",
    title: "Ship login",
    description: "Remote copy",
    status: "todo",
    assignee: null,
    url: "https://linear.app/LIN-1",
    groupKey: "auth",
    groupLabel: "Auth",
    updatedAt: "2026-09-02T00:00:00Z",
    ...overrides,
  };
}

describe("kanban sync", () => {
  it("recommends creating a local card for an unmatched remote issue", () => {
    const recs = reconcileIssues({
      localCards: [],
      remotes: [makeRemote()],
      master: "remote",
    });

    expect(recs).toEqual([
      expect.objectContaining({
        kind: "create_local",
        localCardId: null,
        remote: expect.objectContaining({ externalId: "LIN-1" }),
        recommended: "remote",
      }),
    ]);
  });

  it("recommends creating a remote issue for an unmatched local card", () => {
    const recs = reconcileIssues({
      localCards: [makeCard()],
      remotes: [],
      master: "local",
    });

    expect(recs).toEqual([
      expect.objectContaining({
        kind: "create_remote",
        localCardId: "card-1",
        recommended: "local",
      }),
    ]);
  });

  it("matches by external id and recommends extending local when remote is master", () => {
    const recs = reconcileIssues({
      localCards: [
        makeCard({
          external_id: "LIN-1",
          source_id: "linear-1",
          description: "Local copy",
        }),
      ],
      remotes: [makeRemote({ description: "Remote copy" })],
      master: "remote",
    });

    expect(recs).toHaveLength(1);
    expect(recs[0]?.kind).toBe("extend_local");
    expect(recs[0]?.recommended).toBe("remote");
    expect(recs[0]?.fieldDiffs).toEqual([
      expect.objectContaining({
        field: "description",
        local: "Local copy",
        remote: "Remote copy",
      }),
    ]);
  });

  it("matches by title when ids are missing", () => {
    const recs = reconcileIssues({
      localCards: [makeCard({ title: "Ship Login" })],
      remotes: [makeRemote({ title: "ship login", description: "Local copy" })],
      master: "local",
    });

    expect(recs[0]?.kind).toBe("matched");
    expect(recs[0]?.localCardId).toBe("card-1");
  });

  it("flags a conflict when both sides changed since the last fingerprint", () => {
    const fingerprint = issueFingerprint({
      title: "Ship login",
      status: "todo",
      assignee: null,
      description: "Shared",
    });
    const recs = reconcileIssues({
      localCards: [
        makeCard({
          external_id: "LIN-1",
          source_id: "linear-1",
          description: "Local edit",
          status: "in progress",
        }),
      ],
      remotes: [
        makeRemote({
          description: "Remote edit",
          status: "review",
        }),
      ],
      master: "remote",
      fingerprints: { "linear-1:LIN-1": fingerprint },
    });

    expect(recs[0]?.kind).toBe("conflict");
    expect(recs[0]?.recommended).toBe("remote");
    expect(fingerprintsEqual(fingerprint, fingerprint)).toBe(true);
  });

  it("flags remote-vs-remote conflicts for the same title", () => {
    const recs = reconcileIssues({
      localCards: [
        makeCard({
          external_id: "LIN-1",
          source_id: "linear-1",
        }),
      ],
      remotes: [
        makeRemote({
          sourceId: "linear-1",
          externalId: "LIN-1",
          description: "From Linear",
        }),
        makeRemote({
          sourceId: "jira-1",
          externalId: "JIRA-9",
          description: "From Jira",
          groupKey: "auth",
          groupLabel: "Auth",
        }),
      ],
      master: "local",
    });

    expect(recs.some((rec) => rec.kind === "remote_remote")).toBe(true);
    const remoteRemote = recs.find((rec) => rec.kind === "remote_remote");
    expect(remoteRemote?.otherRemote?.externalId).toBe("JIRA-9");
  });

  it("derives code issues from linked PRs and branches", () => {
    const issues = deriveCodeIssues(
      [
        makeCard({
          id: "pr-card",
          title: "Fix auth",
          linked_pr: "https://github.com/acme/app/pull/12",
          linked_branch: "fix/auth",
        }),
      ],
      "code-1",
    );

    expect(issues).toEqual([
      expect.objectContaining({
        sourceId: "code-1",
        externalId: "https://github.com/acme/app/pull/12",
        groupKey: "fix/auth",
      }),
    ]);
  });

  it("parses a remote issue snapshot from JSON", () => {
    const issues = parseRemoteIssuesJson(
      JSON.stringify([
        {
          id: "ENG-4",
          title: "Add billing",
          status: "In Progress",
          group: "Payments",
        },
      ]),
      "jira-1",
    );

    expect(issues).toEqual([
      expect.objectContaining({
        sourceId: "jira-1",
        externalId: "ENG-4",
        title: "Add billing",
        status: "In Progress",
        groupKey: "Payments",
        groupLabel: "Payments",
      }),
    ]);
  });
});
