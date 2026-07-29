import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { BackendKind } from "#/api/backend-registry/types";
import type { Provider } from "#/types/settings";

export type ConversationSortField = "created" | "updated";
export type ThreadScope = "all" | "relevant";
export type OrganizeMode = "grouped" | "chronological";

/** Max conversations shown under a workspace/repo folder before "View more". */
export const GROUP_CONVERSATIONS_PREVIEW_LIMIT = 5;

interface GroupConversationPreviewOptions {
  limit?: number;
  expanded: boolean;
  activeConversationId?: string | null;
}

export function getGroupConversationPreview(
  conversations: readonly AppConversation[],
  options: GroupConversationPreviewOptions,
): {
  visibleConversations: AppConversation[];
  isPreviewTruncated: boolean;
  isShowingAll: boolean;
} {
  const limit = options.limit ?? GROUP_CONVERSATIONS_PREVIEW_LIMIT;

  if (options.expanded || conversations.length <= limit) {
    return {
      visibleConversations: [...conversations],
      isPreviewTruncated: conversations.length > limit,
      isShowingAll: true,
    };
  }

  const activeIndex =
    options.activeConversationId != null
      ? conversations.findIndex((c) => c.id === options.activeConversationId)
      : -1;

  if (activeIndex >= limit) {
    const activeConversation = conversations[activeIndex];
    return {
      visibleConversations: [
        ...conversations.slice(0, limit - 1),
        activeConversation,
      ],
      isPreviewTruncated: true,
      isShowingAll: false,
    };
  }

  return {
    visibleConversations: conversations.slice(0, limit),
    isPreviewTruncated: conversations.length > limit,
    isShowingAll: false,
  };
}

export function resolvePinnedConversations(
  pinnedIds: readonly string[],
  conversations: readonly AppConversation[],
): AppConversation[] {
  const byId = new Map(
    conversations.map((conversation) => [conversation.id, conversation]),
  );
  return pinnedIds
    .map((id) => byId.get(id))
    .filter(
      (conversation): conversation is AppConversation => conversation != null,
    );
}

export function filterOutPinnedConversations(
  conversations: readonly AppConversation[],
  pinnedIds: readonly string[],
): AppConversation[] {
  if (pinnedIds.length === 0) {
    return [...conversations];
  }

  const pinnedSet = new Set(pinnedIds);
  return conversations.filter(
    (conversation) => !pinnedSet.has(conversation.id),
  );
}

/** Subset of `useCreateConversation` variables for launching from a group row */
export type ConversationGroupLaunch = {
  workingDir?: string;
  repository?: {
    name: string;
    gitProvider: Provider;
    branch?: string;
  };
};

function buildGroupLaunch(
  id: string,
  backendKind: BackendKind,
  conversations: AppConversation[],
): ConversationGroupLaunch {
  if (backendKind === "local") {
    if (id === "__none_workspace") {
      return {};
    }
    if (id.startsWith("ws:")) {
      return { workingDir: id.slice(3) };
    }
    return {};
  }

  if (id === "__none_repo") {
    return {};
  }
  if (id.startsWith("repo:")) {
    const name = id.slice(5);
    const sample = conversations[0];
    const gitProvider = (sample?.git_provider ?? "github") as Provider;
    const branch = sample?.selected_branch ?? "main";
    return {
      repository: {
        name,
        gitProvider,
        branch,
      },
    };
  }

  return {};
}

export function parseConversationTimeMs(iso: string | undefined): number {
  if (!iso) return 0;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : 0;
}

export function sortConversationsByField(
  items: readonly AppConversation[],
  field: ConversationSortField,
): AppConversation[] {
  const key = field === "created" ? "created_at" : "updated_at";
  return [...items].sort(
    (a, b) => parseConversationTimeMs(b[key]) - parseConversationTimeMs(a[key]),
  );
}

function workspaceGroup(conversation: AppConversation): {
  id: string;
  label: string;
} {
  // Group by the user-selected workspace (a stable identifier shared by
  // every conversation launched from the same picker selection), not
  // `workspace.working_dir` — that field holds the per-conversation
  // worktree path the agent-server creates, which is unique per
  // conversation and would fragment the grouping.
  //
  // Normalize first, then check emptiness: inputs like "/", "///", or
  // "   ///" trim+strip to "" and must fall back to the "no workspace"
  // bucket rather than producing a stray `ws:` group with no label.
  const normalized = conversation.selected_workspace
    ?.trim()
    .replace(/\/+$/, "");
  if (!normalized) {
    return { id: "__none_workspace", label: "" };
  }
  const label = normalized.split("/").filter(Boolean).pop() ?? normalized;
  return { id: `ws:${normalized}`, label };
}

function repositoryGroup(conversation: AppConversation): {
  id: string;
  label: string;
} {
  // Mirror `workspaceGroup`'s normalize-then-check order so "/", "///",
  // and trailing-slash variants of the same repo all collapse to one
  // group instead of producing a stray `repo:/` bucket.
  const normalized = conversation.selected_repository
    ?.trim()
    .replace(/\/+$/, "");
  if (!normalized) {
    return { id: "__none_repo", label: "" };
  }
  const parts = normalized.split("/").filter(Boolean);
  const label = parts.length
    ? (parts[parts.length - 1] ?? normalized).replace(/\.git$/, "")
    : normalized.replace(/\.git$/, "");
  return { id: `repo:${normalized}`, label };
}

export function groupConversations(
  items: readonly AppConversation[],
  backendKind: BackendKind,
  sortField: ConversationSortField,
  labels: { emptyWorkspace: string; emptyRepository: string },
): {
  id: string;
  label: string;
  conversations: AppConversation[];
  launch: ConversationGroupLaunch;
}[] {
  const byId = new Map<
    string,
    { label: string; conversations: AppConversation[] }
  >();

  for (const c of items) {
    const { id, label: rawLabel } =
      backendKind === "local" ? workspaceGroup(c) : repositoryGroup(c);
    const label =
      id === "__none_workspace"
        ? labels.emptyWorkspace
        : id === "__none_repo"
          ? labels.emptyRepository
          : rawLabel;
    const bucket = byId.get(id);
    if (bucket) {
      bucket.conversations.push(c);
    } else {
      byId.set(id, { label, conversations: [c] });
    }
  }

  const groups = [...byId.entries()].map(([id, g]) => {
    const conversations = sortConversationsByField(g.conversations, sortField);
    return {
      id,
      label: g.label,
      conversations,
      launch: buildGroupLaunch(id, backendKind, conversations),
    };
  });

  // Use reduce instead of `Math.max(...arr)` — the spread form would push
  // every conversation onto the call stack as a separate argument, which
  // hits JS engines' ~100k-arg limit on very large buckets.
  const groupOrderKey = (g: (typeof groups)[number]) =>
    g.conversations.reduce(
      (max, c) =>
        Math.max(
          max,
          parseConversationTimeMs(
            sortField === "created" ? c.created_at : c.updated_at,
          ),
        ),
      0,
    );

  groups.sort((a, b) => groupOrderKey(b) - groupOrderKey(a));
  return groups;
}

export function applyGroupFolderOrder<T extends { id: string }>(
  groups: readonly T[],
  order: readonly string[],
): T[] {
  if (order.length === 0) {
    return [...groups];
  }

  const byId = new Map(groups.map((group) => [group.id, group]));
  const ordered: T[] = [];
  const seen = new Set<string>();

  for (const id of order) {
    const group = byId.get(id);
    if (group) {
      ordered.push(group);
      seen.add(id);
    }
  }

  for (const group of groups) {
    if (!seen.has(group.id)) {
      ordered.push(group);
    }
  }

  return ordered;
}

export type GroupFolderDropPosition = "before" | "after";

export function moveGroupFolderOrder(
  order: readonly string[],
  groupIds: readonly string[],
  activeGroupId: string,
  targetGroupId: string,
  position: GroupFolderDropPosition = "after",
): string[] {
  if (activeGroupId === targetGroupId) {
    return [...order];
  }

  const effectiveOrder = applyGroupFolderOrder(
    groupIds.map((id) => ({ id })),
    order,
  ).map((group) => group.id);
  const fromIndex = effectiveOrder.indexOf(activeGroupId);
  const toIndex = effectiveOrder.indexOf(targetGroupId);
  if (fromIndex < 0 || toIndex < 0) {
    return [...order];
  }

  const nextOrder = [...effectiveOrder];
  nextOrder.splice(fromIndex, 1);
  const adjustedTargetIndex = nextOrder.indexOf(targetGroupId);
  const insertIndex =
    position === "before" ? adjustedTargetIndex : adjustedTargetIndex + 1;
  nextOrder.splice(insertIndex, 0, activeGroupId);
  return nextOrder;
}
