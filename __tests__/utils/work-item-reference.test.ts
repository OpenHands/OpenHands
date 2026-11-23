import { describe, expect, it } from "vitest";
import {
  WORK_ITEM_ID_TAG_KEY,
  WORK_ITEM_PROVIDER_TAG_KEY,
  WORK_ITEM_REPOSITORY_TAG_KEY,
  WORK_ITEM_TAG_KEYS,
  WORK_ITEM_TAG_VALUE_MAX_LENGTH,
  WORK_ITEM_TITLE_TAG_KEY,
  WORK_ITEM_URL_TAG_KEY,
  buildWorkItemTags,
  formatWorkItemIdentifier,
  formatWorkItemTitle,
  getWorkItemDisplayTitleFromTags,
  getWorkItemProviderPrefix,
  getWorkItemReferenceFromTags,
  hasCompleteWorkItemReference,
  type WorkItemReference,
} from "#/utils/work-item-reference";

describe("work-item reference formatting", () => {
  it("formats a GitHub issue with the GH prefix", () => {
    expect(
      formatWorkItemTitle({
        provider: "github",
        id: "142",
        title: "Mission Provider Filtering",
      }),
    ).toBe("GH-142 \u2014 Mission Provider Filtering");
  });

  it("formats a GitLab issue with the GL prefix", () => {
    expect(
      formatWorkItemTitle({
        provider: "gitlab",
        id: "87",
        title: "Refactor Inventory Service",
      }),
    ).toBe("GL-87 \u2014 Refactor Inventory Service");
  });

  it("formats a Jira work item using its own project key verbatim", () => {
    expect(
      formatWorkItemTitle({
        provider: "jira",
        id: "SYN-241",
        title: "Implement Mission Rewards",
      }),
    ).toBe("SYN-241 \u2014 Implement Mission Rewards");
  });

  it("formats an Azure DevOps work item with the ADO prefix", () => {
    expect(
      formatWorkItemTitle({
        provider: "azure_devops",
        id: "5831",
        title: "Add Player Progression",
      }),
    ).toBe("ADO-5831 \u2014 Add Player Progression");
  });

  it("does not double-prefix an id that already carries a provider key", () => {
    expect(formatWorkItemIdentifier("jira", "SYN-241")).toBe("SYN-241");
    expect(formatWorkItemIdentifier("github", "GH-142")).toBe("GH-142");
  });

  it("never assumes an identifier is numeric", () => {
    expect(formatWorkItemIdentifier("jira", "PROJ-1")).toBe("PROJ-1");
    expect(formatWorkItemIdentifier("github", "abc")).toBe("abc");
  });

  it("preserves special characters and Unicode in the title", () => {
    expect(
      formatWorkItemTitle({
        provider: "github",
        id: "7",
        title: "Fix \u2018caf\u00e9\u2019 r\u00e9sum\u00e9 \ud83d\ude80 <b>&</b>",
      }),
    ).toBe("GH-7 \u2014 Fix \u2018caf\u00e9\u2019 r\u00e9sum\u00e9 \ud83d\ude80 <b>&</b>");
  });

  it("collapses whitespace in a long title without truncating it", () => {
    const title = "A ".repeat(120) + "final";
    const formatted = formatWorkItemTitle({
      provider: "github",
      id: "9",
      title: `  ${title}  `,
    });
    expect(formatted).toBe(`GH-9 \u2014 ${title.replace(/\s+/g, " ")}`);
  });

  it("falls back to the identifier alone when the title is missing", () => {
    expect(formatWorkItemTitle({ provider: "github", id: "5", title: "" })).toBe(
      "GH-5",
    );
    expect(
      formatWorkItemTitle({ provider: "github", id: "5", title: "   " }),
    ).toBe("GH-5");
  });

  it("returns null when the reference or its id is missing", () => {
    expect(formatWorkItemTitle(null)).toBeNull();
    expect(formatWorkItemTitle(undefined)).toBeNull();
    expect(formatWorkItemTitle({ provider: "github", id: "", title: "x" })).toBe(
      null,
    );
    expect(
      formatWorkItemTitle({ provider: "github", id: "   ", title: "x" }),
    ).toBeNull();
  });

  it("normalizes provider prefixes across casing, spacing and punctuation", () => {
    expect(getWorkItemProviderPrefix("GitHub")).toBe("GH");
    expect(getWorkItemProviderPrefix("  gitlab ")).toBe("GL");
    expect(getWorkItemProviderPrefix("Azure DevOps")).toBe("ADO");
    expect(getWorkItemProviderPrefix("azure_devops")).toBe("ADO");
    expect(getWorkItemProviderPrefix("Azure-DevOps")).toBe("ADO");
  });

  it("falls back to an uppercased provider token for unknown providers", () => {
    expect(getWorkItemProviderPrefix("linear")).toBe("LINEAR");
    expect(getWorkItemProviderPrefix("My Tracker")).toBe("MYTRACKER");
    expect(getWorkItemProviderPrefix("")).toBe("");
  });

  it("renders an identifier without a prefix when the provider is empty", () => {
    expect(formatWorkItemIdentifier("", "42")).toBe("42");
    expect(formatWorkItemTitle({ provider: "", id: "42", title: "T" })).toBe(
      "42 \u2014 T",
    );
  });
});

describe("hasCompleteWorkItemReference", () => {
  it("requires a provider and an id, but not a title", () => {
    expect(
      hasCompleteWorkItemReference({ provider: "github", id: "1", title: "" }),
    ).toBe(true);
    expect(
      hasCompleteWorkItemReference({ provider: "", id: "1", title: "t" }),
    ).toBe(false);
    expect(
      hasCompleteWorkItemReference({ provider: "github", id: "", title: "t" }),
    ).toBe(false);
    expect(hasCompleteWorkItemReference(null)).toBe(false);
  });
});

describe("work-item tags round-trip", () => {
  const reference: WorkItemReference = {
    provider: "github",
    id: "142",
    title: "Mission Provider Filtering",
    url: "https://github.com/owner/repo/issues/142",
    repository: "owner/repo",
  };

  it("stamps only lowercase alphanumeric tag keys", () => {
    for (const key of WORK_ITEM_TAG_KEYS) {
      expect(key).toMatch(/^[a-z0-9]+$/);
    }
  });

  it("writes a complete snapshot and reads it back", () => {
    const tags = buildWorkItemTags(reference);
    expect(tags).toEqual({
      [WORK_ITEM_PROVIDER_TAG_KEY]: "github",
      [WORK_ITEM_ID_TAG_KEY]: "142",
      [WORK_ITEM_TITLE_TAG_KEY]: "Mission Provider Filtering",
      [WORK_ITEM_URL_TAG_KEY]: "https://github.com/owner/repo/issues/142",
      [WORK_ITEM_REPOSITORY_TAG_KEY]: "owner/repo",
    });
    expect(getWorkItemReferenceFromTags(tags)).toEqual(reference);
    expect(getWorkItemDisplayTitleFromTags(tags)).toBe(
      "GH-142 \u2014 Mission Provider Filtering",
    );
  });

  it("drops empty and whitespace-only fields instead of writing blank tags", () => {
    const tags = buildWorkItemTags({
      provider: "jira",
      id: "SYN-1",
      title: "   ",
      url: "",
      repository: null,
    });
    expect(tags).toEqual({
      [WORK_ITEM_PROVIDER_TAG_KEY]: "jira",
      [WORK_ITEM_ID_TAG_KEY]: "SYN-1",
    });
    expect(getWorkItemDisplayTitleFromTags(tags)).toBe("SYN-1");
  });

  it("returns no tags for a missing reference", () => {
    expect(buildWorkItemTags(null)).toEqual({});
    expect(buildWorkItemTags(undefined)).toEqual({});
  });

  it("drops a value the agent-server would reject for exceeding the tag limit", () => {
    const tags = buildWorkItemTags({
      provider: "github",
      id: "1",
      title: "x".repeat(WORK_ITEM_TAG_VALUE_MAX_LENGTH + 1),
      url: "https://example.test/".padEnd(WORK_ITEM_TAG_VALUE_MAX_LENGTH, "y"),
    });
    expect(tags).toEqual({
      [WORK_ITEM_PROVIDER_TAG_KEY]: "github",
      [WORK_ITEM_ID_TAG_KEY]: "1",
      [WORK_ITEM_URL_TAG_KEY]: "https://example.test/".padEnd(
        WORK_ITEM_TAG_VALUE_MAX_LENGTH,
        "y",
      ),
    });
  });

  it("degrades gracefully for conversations without a work-item snapshot", () => {
    expect(getWorkItemReferenceFromTags(null)).toBeNull();
    expect(getWorkItemReferenceFromTags(undefined)).toBeNull();
    expect(getWorkItemReferenceFromTags({})).toBeNull();
    expect(getWorkItemReferenceFromTags({ origin: "slack" })).toBeNull();
    expect(getWorkItemDisplayTitleFromTags({})).toBeNull();
    // An id present without a provider still names the conversation.
    expect(
      getWorkItemDisplayTitleFromTags({
        [WORK_ITEM_ID_TAG_KEY]: "42",
        [WORK_ITEM_TITLE_TAG_KEY]: "Untracked",
      }),
    ).toBe("42 \u2014 Untracked");
  });
});
