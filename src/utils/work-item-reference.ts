/**
 * Provider-neutral backlog work-item identity.
 *
 * When an agent is explicitly launched to work on a backlog item from an
 * integrated issue tracker (GitHub issues, GitLab issues, Jira work items,
 * Azure DevOps work items, ...), the resulting conversation should be named
 * after that item so autonomous activity is traceable back to the backlog.
 *
 * This module is the single, provider-neutral home for that concept. It
 * deliberately depends on nothing provider-specific: callers hand it a
 * `WorkItemReference` that was captured at launch time, and it renders the
 * display name and the persistence tags. Keeping the format here (rather than
 * in the conversation builder or the display hook) means the launch path and
 * the display path can never disagree about how a work item is named.
 *
 * A work item is only ever associated *explicitly* by the caller. Nothing in
 * this module inspects prompt text or assumes that a connected repository
 * implies a ticket.
 */

/**
 * A backlog item an agent was explicitly launched against. Provider is an
 * open string (not a closed union) so a new tracker integration can supply a
 * reference without editing this module; the provider prefix lookup falls
 * back to a normalized form of the provider name for anything unknown.
 */
export interface WorkItemReference {
  /** Tracker that owns the item, e.g. "github", "gitlab", "jira", "azure_devops". */
  provider: string;
  /**
   * The provider's own identifier, verbatim. A GitHub issue number ("142"), a
   * Jira issue key ("SYN-241") and an Azure DevOps work-item id ("5831") are
   * all valid. Never assume an id is numeric.
   */
  id: string;
  /** Human-readable item title. Optional; an identifier-only name is still valid. */
  title: string;
  /** Canonical link to the item, when the integration reports one. */
  url?: string | null;
  /** The repository the item belongs to ("owner/repo"), when known. */
  repository?: string | null;
}

/**
 * Conversation tag keys that persist the launch-time snapshot of the work
 * item. Keys must match the agent-server's `^[a-z0-9]+$` tag-key rule.
 *
 * The snapshot is deliberate: an issue title edited after launch must not
 * rename an existing conversation, so the display name is fixed at launch.
 */
export const WORK_ITEM_PROVIDER_TAG_KEY = "workitemprovider";
export const WORK_ITEM_ID_TAG_KEY = "workitemid";
export const WORK_ITEM_TITLE_TAG_KEY = "workitemtitle";
export const WORK_ITEM_URL_TAG_KEY = "workitemurl";
export const WORK_ITEM_REPOSITORY_TAG_KEY = "workitemrepository";

/** Every tag key this module writes, for the reserved-key set. */
export const WORK_ITEM_TAG_KEYS: readonly string[] = [
  WORK_ITEM_PROVIDER_TAG_KEY,
  WORK_ITEM_ID_TAG_KEY,
  WORK_ITEM_TITLE_TAG_KEY,
  WORK_ITEM_URL_TAG_KEY,
  WORK_ITEM_REPOSITORY_TAG_KEY,
];

/** Separator between the identifier and the title in the display name. */
export const WORK_ITEM_TITLE_SEPARATOR = "\u2014";

/**
 * The agent-server rejects a tag value longer than this (`TAG_VALUE_MAX_LENGTH`
 * in the SDK), which would block the conversation from starting. Tag values
 * are capped so an unusually long title or URL degrades to a shorter snapshot
 * rather than failing the launch; the full name still rides the server title.
 */
export const WORK_ITEM_TAG_VALUE_MAX_LENGTH = 256;

/**
 * Canonical display prefixes, keyed by the *normalized* provider token
 * (lowercased, non-alphanumerics stripped). Centralized here so every tracker
 * renders consistently; extend by adding an entry, not by branching on a
 * provider name at a call site. Never hardcode a project key (a Jira key such
 * as "SYN-241" already carries its own project prefix and is used verbatim).
 */
export const WORK_ITEM_PROVIDER_PREFIXES: Readonly<Record<string, string>> = {
  github: "GH",
  gitlab: "GL",
  jira: "JIRA",
  azuredevops: "ADO",
  bitbucket: "BB",
  bitbucketdatacenter: "BBDC",
  forgejo: "FJ",
};

/** Lowercase, strip anything that is not a letter or digit. */
function normalizeProviderToken(provider: string): string {
  return provider
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

/**
 * The display prefix for a provider, normalized so "GitHub", "github" and
 * " gh " all resolve to "GH". An unknown provider yields the normalized,
 * uppercased provider name (e.g. "linear" -> "LINEAR"); an empty provider
 * yields "" so the identifier stands alone.
 */
export function getWorkItemProviderPrefix(provider: string): string {
  const token = normalizeProviderToken(provider);
  if (!token) return "";
  return WORK_ITEM_PROVIDER_PREFIXES[token] ?? token.toUpperCase();
}

const NUMERIC_ID = /^\d+$/;

/**
 * The provider-qualified identifier shown in a display name.
 *
 * A purely numeric id is prefixed with the provider's canonical prefix
 * ("142" -> "GH-142"). An id that already carries its own key — a Jira issue
 * key like "SYN-241", or any non-numeric id — is used verbatim, so no project
 * key is ever invented. Returns null when there is no usable id.
 */
export function formatWorkItemIdentifier(
  provider: string,
  id: string | null | undefined,
): string | null {
  const trimmed = id?.trim();
  if (!trimmed) return null;
  if (!NUMERIC_ID.test(trimmed)) return trimmed;
  const prefix = getWorkItemProviderPrefix(provider);
  return prefix ? `${prefix}-${trimmed}` : trimmed;
}

/**
 * The full display name for a work item: `[PREFIX]-[ID] — [TITLE]`.
 *
 * Returns null when the reference carries no usable id. A reference with an id
 * but no title yields the identifier alone, so an incomplete title degrades to
 * a stable, LLM-free name rather than failing. Whitespace in the title is
 * collapsed; the title is otherwise preserved verbatim (Unicode included).
 */
export function formatWorkItemTitle(
  reference: WorkItemReference | null | undefined,
): string | null {
  if (!reference) return null;
  const identifier = formatWorkItemIdentifier(reference.provider, reference.id);
  if (!identifier) return null;
  const title = reference.title?.trim().replace(/\s+/g, " ");
  return title
    ? `${identifier} ${WORK_ITEM_TITLE_SEPARATOR} ${title}`
    : identifier;
}

/**
 * Whether a reference is complete enough to name a conversation without an
 * LLM: it must identify the item (provider and id). The title is optional —
 * the identifier alone is a valid name.
 */
export function hasCompleteWorkItemReference(
  reference: WorkItemReference | null | undefined,
): boolean {
  return (
    !!reference &&
    normalizeProviderToken(reference.provider).length > 0 &&
    !!reference.id?.trim()
  );
}

/**
 * The launch-time snapshot as conversation tags. Empty or whitespace-only
 * fields are dropped so a partial reference never writes a blank tag, and any
 * value over {@link WORK_ITEM_TAG_VALUE_MAX_LENGTH} is dropped too — the
 * agent-server would reject the whole conversation otherwise. Only
 * non-sensitive identity is stored — never tokens or internal row ids.
 */
export function buildWorkItemTags(
  reference: WorkItemReference | null | undefined,
): Record<string, string> {
  if (!reference) return {};
  const tags: Record<string, string> = {};
  const set = (key: string, value: string | null | undefined) => {
    const trimmed = value?.trim();
    if (trimmed && trimmed.length <= WORK_ITEM_TAG_VALUE_MAX_LENGTH) {
      tags[key] = trimmed;
    }
  };
  set(WORK_ITEM_PROVIDER_TAG_KEY, reference.provider);
  set(WORK_ITEM_ID_TAG_KEY, reference.id);
  set(WORK_ITEM_TITLE_TAG_KEY, reference.title);
  set(WORK_ITEM_URL_TAG_KEY, reference.url);
  set(WORK_ITEM_REPOSITORY_TAG_KEY, reference.repository);
  return tags;
}

/**
 * Reconstruct the snapshot from a conversation's tags. Returns null when the
 * tags carry no identifier, so conversations created before this feature (or
 * without a work item) are unaffected. A malformed tag value is treated as
 * absent rather than throwing.
 */
export function getWorkItemReferenceFromTags(
  tags: Record<string, string> | null | undefined,
): WorkItemReference | null {
  if (!tags) return null;
  const id = tags[WORK_ITEM_ID_TAG_KEY]?.trim();
  if (!id) return null;
  return {
    provider: tags[WORK_ITEM_PROVIDER_TAG_KEY]?.trim() ?? "",
    id,
    title: tags[WORK_ITEM_TITLE_TAG_KEY]?.trim() ?? "",
    url: tags[WORK_ITEM_URL_TAG_KEY]?.trim() || null,
    repository: tags[WORK_ITEM_REPOSITORY_TAG_KEY]?.trim() || null,
  };
}

/**
 * The display name for a conversation's work item, read from its tags.
 * Convenience over {@link getWorkItemReferenceFromTags} +
 * {@link formatWorkItemTitle} for the display path.
 */
export function getWorkItemDisplayTitleFromTags(
  tags: Record<string, string> | null | undefined,
): string | null {
  return formatWorkItemTitle(getWorkItemReferenceFromTags(tags));
}
