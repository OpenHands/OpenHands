import type {
  KanbanCard,
  UpdateCardPayload,
} from "#/api/kanban-service/kanban-types";

export const KANBAN_SYNC_STORAGE_KEY = "openhands-kanban-sync";

export const KANBAN_SOURCE_PROVIDERS = [
  "jira",
  "linear",
  "plane",
  "github",
  "code",
] as const;

export const KANBAN_MASTER_SIDES = ["local", "remote"] as const;

export const KANBAN_MERGE_KINDS = [
  "create_local",
  "create_remote",
  "extend_local",
  "extend_remote",
  "conflict",
  "remote_remote",
  "matched",
] as const;

export const KANBAN_LANE_KINDS = ["manual", "remote"] as const;

export type KanbanSourceProvider = (typeof KANBAN_SOURCE_PROVIDERS)[number];
export type KanbanMasterSide = (typeof KANBAN_MASTER_SIDES)[number];
export type KanbanMergeKind = (typeof KANBAN_MERGE_KINDS)[number];
export type KanbanLaneKind = (typeof KANBAN_LANE_KINDS)[number];
export interface KanbanLane {
  id: string;
  boardId: string;
  parentId: string | null;
  name: string;
  kind: KanbanLaneKind;
  sourceId: string | null;
  remoteKey: string | null;
  position: number;
}

export interface KanbanMappedSource {
  id: string;
  boardId: string;
  provider: KanbanSourceProvider;
  label: string;
  url: string | null;
  master: KanbanMasterSide;
  createdAt: string;
}

export interface RemoteIssue {
  sourceId: string;
  externalId: string;
  title: string;
  description: string | null;
  status: string;
  assignee: string | null;
  url: string | null;
  groupKey: string | null;
  groupLabel: string | null;
  updatedAt: string;
}

export interface IssueFingerprint {
  title: string;
  status: string;
  assignee: string;
  description: string;
}

export interface MergeFieldDiff {
  field: "title" | "status" | "assignee" | "description";
  local: string;
  remote: string;
}

export interface MergeRecommendation {
  id: string;
  kind: KanbanMergeKind;
  boardId: string | null;
  localCardId: string | null;
  remote: RemoteIssue | null;
  otherRemote: RemoteIssue | null;
  recommended: KanbanMasterSide;
  fieldDiffs: MergeFieldDiff[];
}

export function normalizeIssueTitle(title: string): string {
  return title.trim().toLowerCase().replace(/\s+/g, " ");
}

export function fingerprintKey(sourceId: string, externalId: string): string {
  return `${sourceId}:${externalId}`;
}

export function issueFingerprint(fields: {
  title: string;
  status: string;
  assignee: string | null;
  description: string | null;
}): IssueFingerprint {
  return {
    title: normalizeIssueTitle(fields.title),
    status: (fields.status ?? "").trim().toLowerCase(),
    assignee: (fields.assignee ?? "").trim(),
    description: (fields.description ?? "").trim(),
  };
}

export function fingerprintsEqual(
  left: IssueFingerprint,
  right: IssueFingerprint,
): boolean {
  return (
    left.title === right.title &&
    left.status === right.status &&
    left.assignee === right.assignee &&
    left.description === right.description
  );
}

export function fieldDiffs(
  local: IssueFingerprint,
  remote: IssueFingerprint,
): MergeFieldDiff[] {
  const fields: MergeFieldDiff["field"][] = [
    "title",
    "status",
    "assignee",
    "description",
  ];
  return fields
    .filter((field) => local[field] !== remote[field])
    .map((field) => ({
      field,
      local: local[field],
      remote: remote[field],
    }));
}

export function cardFingerprint(card: KanbanCard): IssueFingerprint {
  return issueFingerprint({
    title: card.title,
    status: card.status,
    assignee: card.assignee,
    description: card.description,
  });
}

export function remoteFingerprint(issue: RemoteIssue): IssueFingerprint {
  return issueFingerprint(issue);
}

function recommendationId(
  kind: KanbanMergeKind,
  localCardId: string | null,
  remote: RemoteIssue | null,
  otherRemote: RemoteIssue | null,
): string {
  return [
    kind,
    localCardId ?? "none",
    remote ? fingerprintKey(remote.sourceId, remote.externalId) : "none",
    otherRemote
      ? fingerprintKey(otherRemote.sourceId, otherRemote.externalId)
      : "none",
  ].join(":");
}

function matchLocalCard(
  remotes: RemoteIssue,
  cards: KanbanCard[],
  claimed: Set<string>,
): KanbanCard | null {
  const byId = cards.find(
    (card) =>
      !claimed.has(card.id) &&
      card.source_id === remotes.sourceId &&
      card.external_id === remotes.externalId,
  );
  if (byId) return byId;

  const needle = normalizeIssueTitle(remotes.title);
  const titled = cards.filter(
    (card) =>
      !claimed.has(card.id) && normalizeIssueTitle(card.title) === needle,
  );
  return titled.length === 1 ? titled[0]! : null;
}

export function reconcileIssues(input: {
  localCards: KanbanCard[];
  remotes: RemoteIssue[];
  master: KanbanMasterSide;
  fingerprints?: Record<string, IssueFingerprint>;
  boardId?: string | null;
}): MergeRecommendation[] {
  const fingerprints = input.fingerprints ?? {};
  const claimed = new Set<string>();
  const recs: MergeRecommendation[] = [];
  const boardId = input.boardId ?? input.localCards[0]?.board_id ?? null;

  const push = (
    kind: KanbanMergeKind,
    localCard: KanbanCard | null,
    remote: RemoteIssue | null,
    otherRemote: RemoteIssue | null,
    diffs: MergeFieldDiff[],
    recommended: KanbanMasterSide = input.master,
  ) => {
    recs.push({
      id: recommendationId(kind, localCard?.id ?? null, remote, otherRemote),
      kind,
      boardId: localCard?.board_id ?? boardId,
      localCardId: localCard?.id ?? null,
      remote,
      otherRemote,
      recommended,
      fieldDiffs: diffs,
    });
  };

  for (const remote of input.remotes) {
    const local = matchLocalCard(remote, input.localCards, claimed);
    if (!local) {
      push("create_local", null, remote, null, []);
      continue;
    }
    claimed.add(local.id);
    const localFp = cardFingerprint(local);
    const remoteFp = remoteFingerprint(remote);
    const diffs = fieldDiffs(localFp, remoteFp);
    const key = fingerprintKey(remote.sourceId, remote.externalId);
    const previous = fingerprints[key];

    if (diffs.length === 0) {
      push("matched", local, remote, null, []);
      continue;
    }

    if (previous) {
      const localChanged = !fingerprintsEqual(localFp, previous);
      const remoteChanged = !fingerprintsEqual(remoteFp, previous);
      if (localChanged && remoteChanged) {
        push("conflict", local, remote, null, diffs);
        continue;
      }
      if (remoteChanged) {
        push("extend_local", local, remote, null, diffs, "remote");
        continue;
      }
      if (localChanged) {
        push("extend_remote", local, remote, null, diffs, "local");
        continue;
      }
    }

    if (input.master === "remote") {
      push("extend_local", local, remote, null, diffs, "remote");
    } else {
      push("extend_remote", local, remote, null, diffs, "local");
    }
  }

  for (const local of input.localCards) {
    if (claimed.has(local.id)) continue;
    push("create_remote", local, null, null, []);
  }

  const remotesByTitle = new Map<string, RemoteIssue[]>();
  for (const remote of input.remotes) {
    const key = normalizeIssueTitle(remote.title);
    const group = remotesByTitle.get(key) ?? [];
    group.push(remote);
    remotesByTitle.set(key, group);
  }
  for (const group of remotesByTitle.values()) {
    const uniqueSources = new Set(group.map((item) => item.sourceId));
    if (uniqueSources.size < 2) continue;
    const [left, right] = group;
    if (!left || !right) continue;
    const diffs = fieldDiffs(remoteFingerprint(left), remoteFingerprint(right));
    if (diffs.length === 0) continue;
    const local = input.localCards.find(
      (card) =>
        card.external_id === left.externalId ||
        card.external_id === right.externalId ||
        normalizeIssueTitle(card.title) === normalizeIssueTitle(left.title),
    );
    push("remote_remote", local ?? null, left, right, diffs);
  }

  return recs;
}

export function deriveCodeIssues(
  cards: KanbanCard[],
  sourceId: string,
): RemoteIssue[] {
  return cards
    .filter((card) => Boolean(card.linked_pr || card.linked_branch))
    .map((card) => ({
      sourceId,
      externalId:
        card.external_id ?? card.linked_pr ?? card.linked_branch ?? card.id,
      title: card.title,
      description: card.description,
      status: card.status,
      assignee: card.assignee,
      url: card.linked_pr,
      groupKey: card.linked_branch,
      groupLabel: card.linked_branch,
      updatedAt: card.updated_at,
    }));
}

interface LooseRemoteIssue {
  externalId?: unknown;
  id?: unknown;
  key?: unknown;
  title?: unknown;
  description?: unknown;
  status?: unknown;
  assignee?: unknown;
  url?: unknown;
  groupKey?: unknown;
  group?: unknown;
  groupLabel?: unknown;
  updatedAt?: unknown;
}

function asString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function parseRemoteIssuesJson(
  raw: string,
  sourceId: string,
): RemoteIssue[] {
  const trimmed = raw.trim();
  if (!trimmed) return [];
  const parsed: unknown = JSON.parse(trimmed);
  let items: unknown[] | null = null;
  if (Array.isArray(parsed)) {
    items = parsed;
  } else if (parsed && typeof parsed === "object") {
    const record = parsed as { issues?: unknown; items?: unknown };
    if (Array.isArray(record.issues)) items = record.issues;
    else if (Array.isArray(record.items)) items = record.items;
  }
  if (!items) return [];
  return items.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const row = item as LooseRemoteIssue;
    const title = asString(row.title);
    const externalId =
      asString(row.externalId) ?? asString(row.id) ?? asString(row.key);
    if (!title || !externalId) return [];
    const group =
      asString(row.groupKey) ?? asString(row.group) ?? asString(row.groupLabel);
    return [
      {
        sourceId,
        externalId,
        title,
        description: asString(row.description),
        status: asString(row.status) ?? "todo",
        assignee: asString(row.assignee),
        url: asString(row.url),
        groupKey: group,
        groupLabel: group,
        updatedAt: asString(row.updatedAt) ?? new Date().toISOString(),
      },
    ];
  });
}

export function lanesFromRemoteIssues(
  boardId: string,
  sourceId: string,
  issues: RemoteIssue[],
  existing: KanbanLane[],
): KanbanLane[] {
  const seen = new Set<string>();
  const next = [...existing];
  let position = existing.reduce(
    (max, lane) => Math.max(max, lane.position + 1),
    0,
  );
  for (const issue of issues) {
    if (!issue.groupKey || seen.has(issue.groupKey)) continue;
    seen.add(issue.groupKey);
    const already = next.find(
      (lane) =>
        lane.boardId === boardId &&
        lane.sourceId === sourceId &&
        lane.remoteKey === issue.groupKey,
    );
    if (already) continue;
    next.push({
      id: crypto.randomUUID(),
      boardId,
      parentId: null,
      name: issue.groupLabel ?? issue.groupKey,
      kind: "remote",
      sourceId,
      remoteKey: issue.groupKey,
      position,
    });
    position += 1;
  }
  return next;
}

export function patchFromRemote(remote: RemoteIssue): UpdateCardPayload {
  return {
    title: remote.title,
    description: remote.description,
    status: remote.status,
    assignee: remote.assignee,
    linked_pr: remote.url,
    origin: "remote",
    external_id: remote.externalId,
    source_id: remote.sourceId,
  };
}

export function openReconcileItems(
  recs: MergeRecommendation[],
): MergeRecommendation[] {
  return recs.filter((rec) => rec.kind !== "matched");
}

export function columnIdForRemoteStatus(
  columns: { id: string; name: string }[],
  status: string | null | undefined,
): string | null {
  if (columns.length === 0) return null;
  const raw = (status ?? "").trim().toLowerCase();
  const aliases: Record<string, string> = {
    todo: "backlog",
    backlog: "backlog",
    "in progress": "in progress",
    doing: "in progress",
    started: "in progress",
    review: "review",
    "in review": "review",
    done: "done",
    complete: "done",
    completed: "done",
  };
  const target = aliases[raw] ?? raw;
  const match = columns.find(
    (column) =>
      column.name.trim().toLowerCase().replace(/\s+/g, " ") === target,
  );
  return match?.id ?? columns[0]?.id ?? null;
}
