/**
 * Pinable rows in the conversation overview panel.
 * Persistence uses the inverse denylist `unpinnedOverviewSections`
 * (empty = all pinned), matching drawer tab pin semantics.
 *
 * Git is a pinable parent section with its own sub-part denylist
 * (`unpinnedOverviewGitParts`), including Changes.
 */
export const CONVERSATION_OVERVIEW_SECTION = {
  workspace: "workspace",
  git: "git",
} as const;

export type ConversationOverviewSection =
  (typeof CONVERSATION_OVERVIEW_SECTION)[keyof typeof CONVERSATION_OVERVIEW_SECTION];

export const CONVERSATION_OVERVIEW_SECTIONS: readonly ConversationOverviewSection[] =
  [CONVERSATION_OVERVIEW_SECTION.workspace, CONVERSATION_OVERVIEW_SECTION.git];

export const VALID_CONVERSATION_OVERVIEW_SECTIONS: ReadonlySet<string> =
  new Set(CONVERSATION_OVERVIEW_SECTIONS);

/**
 * Sub-rows inside the git overview block. Controlled independently via
 * `unpinnedOverviewGitParts` when the parent git section is pinned.
 * Changes lives here (not as a top-level overview section).
 */
export const CONVERSATION_OVERVIEW_GIT_PART = {
  changes: "changes",
  repository: "repository",
  branch: "branch",
  commits: "commits",
  pull_requests: "pull_requests",
} as const;

export type ConversationOverviewGitPart =
  (typeof CONVERSATION_OVERVIEW_GIT_PART)[keyof typeof CONVERSATION_OVERVIEW_GIT_PART];

export const CONVERSATION_OVERVIEW_GIT_PARTS: readonly ConversationOverviewGitPart[] =
  [
    CONVERSATION_OVERVIEW_GIT_PART.changes,
    CONVERSATION_OVERVIEW_GIT_PART.repository,
    CONVERSATION_OVERVIEW_GIT_PART.branch,
    CONVERSATION_OVERVIEW_GIT_PART.commits,
    CONVERSATION_OVERVIEW_GIT_PART.pull_requests,
  ];

export const VALID_CONVERSATION_OVERVIEW_GIT_PARTS: ReadonlySet<string> =
  new Set(CONVERSATION_OVERVIEW_GIT_PARTS);

/** No overview sections are hidden by default. */
export const DEFAULT_UNPINNED_OVERVIEW_SECTIONS: readonly ConversationOverviewSection[] =
  [];

export const DEFAULT_UNPINNED_OVERVIEW_GIT_PARTS: readonly ConversationOverviewGitPart[] =
  [];

/**
 * Visual groups in the panel. Dividers render between non-empty groups.
 * Workspace stands alone below the untitled Git block (Changes + metadata)
 * that the panel prepends when pinned.
 */
export const CONVERSATION_OVERVIEW_SECTION_GROUPS: readonly {
  sections: readonly ConversationOverviewSection[];
}[] = [
  {
    sections: [CONVERSATION_OVERVIEW_SECTION.workspace],
  },
];

export function isOverviewSectionPinned(
  section: ConversationOverviewSection,
  unpinnedOverviewSections: readonly string[],
): boolean {
  return !unpinnedOverviewSections.includes(section);
}

export function isOverviewGitPartPinned(
  part: ConversationOverviewGitPart,
  unpinnedOverviewGitParts: readonly string[],
): boolean {
  return !unpinnedOverviewGitParts.includes(part);
}
