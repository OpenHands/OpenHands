import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { BackendKind } from "#/api/backend-registry/types";
import type { Provider } from "#/types/settings";
import {
  AUTOMATION_NAME_TAG_KEY,
  AUTOMATION_TAG_KEYS,
} from "#/api/agent-server-adapter";

export type ConversationSortField = "created" | "updated";
export type ThreadScope = "all" | "relevant";
export type OrganizeMode = "grouped" | "chronological";
export type AutomationFilterMode =
  | "all"
  | "hide-automations"
  | "only-automations";

/** Max conversations shown under a workspace/repo folder before "View more". */
export const GROUP_CONVERSATIONS_PREVIEW_LIMIT = 5;

interface GroupConversationPreviewOptions {
  limit?: number;
  expanded: boolean;
  activeConversationId?: string | null;
  /**
   * When set, the collapsed preview is drawn only from these conversation IDs
   * (plus the active conversation when it belongs to the group). Expanding
   * still reveals every loaded conversation in `conversations`.
   */
  discoveryConversationIds?: ReadonlySet<string>;
}

/**
 * Builds the frozen collapsed-preview pool for a folder.
 *
 * Global "Load more" may fetch later pages that add conversations to an
 * already-visible folder; those rows stay out of the collapsed preview so the
 * folder's first impression stays stable until the user expands it.
 */
function resolveCollapsedPreviewPool(
  conversations: readonly AppConversation[],
  discoveryConversationIds: ReadonlySet<string> | undefined,
  activeConversationId: string | null | undefined,
): AppConversation[] {
  if (!discoveryConversationIds) {
    return [...conversations];
  }

  const pool = conversations.filter((conversation) =>
    discoveryConversationIds.has(conversation.id),
  );

  // The load-bearing active-conversation guarantee lives in
  // `getGroupDiscoveryConversationIds` (`forceIncludeConversationId`); this
  // fallback only covers callers that pass a discovery set built without it.
  if (
    activeConversationId != null &&
    !pool.some((conversation) => conversation.id === activeConversationId)
  ) {
    const activeConversation = conversations.find(
      (conversation) => conversation.id === activeConversationId,
    );
    if (activeConversation) {
      pool.push(activeConversation);
    }
  }

  return pool;
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
  const pool = resolveCollapsedPreviewPool(
    conversations,
    options.discoveryConversationIds,
    options.activeConversationId,
  );

  let collapsedVisible: AppConversation[];
  if (pool.length <= limit) {
    collapsedVisible = pool;
  } else {
    const activeIndex =
      options.activeConversationId != null
        ? pool.findIndex(
            (conversation) => conversation.id === options.activeConversationId,
          )
        : -1;

    if (activeIndex >= limit) {
      collapsedVisible = [...pool.slice(0, limit - 1), pool[activeIndex]];
    } else {
      collapsedVisible = pool.slice(0, limit);
    }
  }

  const collapsedHidesSomething =
    collapsedVisible.length < conversations.length;

  if (options.expanded) {
    return {
      visibleConversations: [...conversations],
      isPreviewTruncated: collapsedHidesSomething,
      isShowingAll: true,
    };
  }

  return {
    visibleConversations: collapsedVisible,
    isPreviewTruncated: collapsedHidesSomething,
    isShowingAll: !collapsedHidesSomething,
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

/**
 * Facet bucket for automation-born conversations that carry no
 * `automationname` tag (double-underscore prefix mirrors the
 * `__none_workspace` group-id idiom).
 */
export const UNNAMED_AUTOMATION_FACET = "__unnamed__";

/**
 * Whether a conversation was created by an automation run: the cloud backend
 * stamps `trigger: "automation"`, while local agent-server conversations are
 * recognized by the automation tags the SDK workspace attaches at creation.
 */
export function isAutomationConversation(
  conversation: AppConversation,
): boolean {
  if (conversation.trigger === "automation") {
    return true;
  }
  const tags = conversation.tags;
  if (!tags) {
    return false;
  }
  return AUTOMATION_TAG_KEYS.some((key) => Boolean(tags[key]));
}

export function getAutomationNameFacet(conversation: AppConversation): string {
  const name = conversation.tags?.[AUTOMATION_NAME_TAG_KEY]?.trim();
  return name || UNNAMED_AUTOMATION_FACET;
}

/**
 * Unique automation names among the given conversations, sorted for stable
 * menu order; the unnamed bucket (if present) is appended last.
 */
export function collectAutomationNameFacets(
  conversations: readonly AppConversation[],
): string[] {
  const names = new Set<string>();
  let hasUnnamed = false;
  for (const conversation of conversations) {
    if (!isAutomationConversation(conversation)) {
      continue;
    }
    const facet = getAutomationNameFacet(conversation);
    if (facet === UNNAMED_AUTOMATION_FACET) {
      hasUnnamed = true;
    } else {
      names.add(facet);
    }
  }
  const sorted = [...names].sort((a, b) => a.localeCompare(b));
  if (hasUnnamed) {
    sorted.push(UNNAMED_AUTOMATION_FACET);
  }
  return sorted;
}

export function applyAutomationConversationFilter(
  conversations: readonly AppConversation[],
  mode: AutomationFilterMode,
  selectedNames: readonly string[],
  availableFacets: readonly string[],
): AppConversation[] {
  if (mode === "all") {
    return [...conversations];
  }
  if (mode === "hide-automations") {
    return conversations.filter(
      (conversation) => !isAutomationConversation(conversation),
    );
  }
  // "only-automations": an empty selection means every automation run, and a
  // selection that no longer intersects the available facets (stale names
  // persisted from another backend, renamed automations) degrades the same
  // way instead of yielding an unfillable empty list.
  const facetSet = new Set(availableFacets);
  const effectiveNames = selectedNames.filter((name) => facetSet.has(name));
  const narrowing = effectiveNames.length > 0 ? new Set(effectiveNames) : null;
  return conversations.filter((conversation) => {
    if (!isAutomationConversation(conversation)) {
      return false;
    }
    return (
      narrowing === null || narrowing.has(getAutomationNameFacet(conversation))
    );
  });
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

/**
 * Resolves the stable folder identity used by grouped conversation views.
 *
 * Keeping this shared with `groupConversations` lets pagination reason about
 * folder discovery without duplicating workspace/repository normalization.
 */
function getConversationGroupIdentity(
  conversation: AppConversation,
  backendKind: BackendKind,
): { id: string; label: string } {
  return backendKind === "local"
    ? workspaceGroup(conversation)
    : repositoryGroup(conversation);
}

/**
 * Max backend pages fetched for a single grouped "Load more" click. The
 * driver walks past pages that only deepen already-visible folders looking
 * for a new folder, and this cap prevents one click from walking the entire
 * remaining cursor when no undiscovered folder exists. Chronological mode is
 * not capped — it keeps its pre-existing fetch-until-visible behavior.
 */
export const MAX_PAGES_PER_LOAD_MORE_CLICK = 3;

/**
 * Conversation IDs that belong on each folder's discovery page — the first
 * backend page where that folder appeared.
 *
 * Global "Load more" still discovers folders from later pages, but the
 * collapsed preview for an already-visible folder stays frozen to this set.
 * Expanding the folder reads the full grouped `conversations` array instead.
 * `forceIncludeConversationId` keeps the active thread in the preview even
 * when it landed on a non-discovery page.
 */
export function getGroupDiscoveryConversationIds(
  items: readonly AppConversation[],
  pageByConversationId: ReadonlyMap<string, number>,
  backendKind: BackendKind,
  options?: { forceIncludeConversationId?: string | null },
): Set<string> {
  // Resolve each conversation's folder identity exactly once; both passes
  // below (finding each folder's discovery page, then collecting the ids on
  // that page) read from this record instead of re-running the
  // workspace/repository normalization.
  const resolved = items.map((conversation) => ({
    conversationId: conversation.id,
    groupId: getConversationGroupIdentity(conversation, backendKind).id,
    page: pageByConversationId.get(conversation.id) ?? 0,
  }));

  const discoveryPageByGroupId = new Map<string, number>();
  for (const { groupId, page } of resolved) {
    const currentDiscoveryPage = discoveryPageByGroupId.get(groupId);
    if (currentDiscoveryPage === undefined || page < currentDiscoveryPage) {
      discoveryPageByGroupId.set(groupId, page);
    }
  }

  const discoveryIds = new Set<string>();
  for (const { conversationId, groupId, page } of resolved) {
    if (page === discoveryPageByGroupId.get(groupId)) {
      discoveryIds.add(conversationId);
    }
  }

  const forceIncludeId = options?.forceIncludeConversationId;
  if (
    forceIncludeId != null &&
    items.some((conversation) => conversation.id === forceIncludeId)
  ) {
    discoveryIds.add(forceIncludeId);
  }

  return discoveryIds;
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
    const { id, label: rawLabel } = getConversationGroupIdentity(
      c,
      backendKind,
    );
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
