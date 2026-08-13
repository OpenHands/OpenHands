import { describe, expect, it } from "vitest";
import {
  computeVisibleTagChipCount,
  formatConversationTagTooltip,
  getConversationTagLabel,
  getConversationTagLabelKind,
  humanizeConversationTagKey,
  TAG_CHIP_GAP_PX,
  TAG_CHIP_OVERFLOW_WIDTH_PX,
  TAG_CHIP_VALUE_MAX_LENGTH,
  truncateTagChipValue,
} from "#/components/features/conversation-panel/conversation-card/conversation-tag-display";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import { I18nKey } from "#/i18n/declaration";

describe("truncateTagChipValue", () => {
  it("leaves short values unchanged", () => {
    expect(truncateTagChipValue("slack")).toBe("slack");
    expect(truncateTagChipValue("a".repeat(TAG_CHIP_VALUE_MAX_LENGTH))).toBe(
      "a".repeat(TAG_CHIP_VALUE_MAX_LENGTH),
    );
  });

  it("hard-truncates long values with an ellipsis within the budget", () => {
    const value = "a".repeat(TAG_CHIP_VALUE_MAX_LENGTH + 8);
    const truncated = truncateTagChipValue(value);
    expect(truncated).toHaveLength(TAG_CHIP_VALUE_MAX_LENGTH);
    expect(truncated.endsWith("…")).toBe(true);
    expect(
      truncated.startsWith("a".repeat(TAG_CHIP_VALUE_MAX_LENGTH - 1)),
    ).toBe(true);
  });

  it("never splits a surrogate pair at the cut point", () => {
    // Slack-/Discord-stamped values carry emoji; slicing by UTF-16 code unit
    // would leave a lone surrogate that renders as a replacement glyph.
    const value = `${"a".repeat(TAG_CHIP_VALUE_MAX_LENGTH - 1)}🎉extra`;
    const truncated = truncateTagChipValue(value);

    expect(truncated).toBe(`${"a".repeat(TAG_CHIP_VALUE_MAX_LENGTH - 1)}…`);
    for (const codePoint of truncated) {
      expect(codePoint.codePointAt(0)).toBeLessThan(0xd800);
    }
  });

  it("counts astral characters as one character each", () => {
    // 13 emoji + one more: 14 code points fits the budget untouched.
    const value = "🎉".repeat(TAG_CHIP_VALUE_MAX_LENGTH);
    expect(truncateTagChipValue(value)).toBe(value);
    expect(truncateTagChipValue(`${value}🎉`)).toBe(
      `${"🎉".repeat(TAG_CHIP_VALUE_MAX_LENGTH - 1)}…`,
    );
  });
});

describe("computeVisibleTagChipCount", () => {
  it("shows every chip when the container has not been measured yet", () => {
    expect(computeVisibleTagChipCount([40, 40, 40], 0)).toBe(3);
  });

  it("fits as many chips as the single row allows, reserving overflow space", () => {
    // Two 40px chips + gap + overflow reserve must fit; a third does not.
    const widths = [40, 40, 40];
    const forTwo =
      40 + TAG_CHIP_GAP_PX + 40 + TAG_CHIP_GAP_PX + TAG_CHIP_OVERFLOW_WIDTH_PX;
    expect(computeVisibleTagChipCount(widths, forTwo)).toBe(2);
  });

  it("drops the overflow reserve when every chip fits", () => {
    const widths = [40, 40];
    const exact = 40 + TAG_CHIP_GAP_PX + 40;
    expect(computeVisibleTagChipCount(widths, exact)).toBe(2);
  });

  it("returns 0 when even one chip plus overflow cannot fit", () => {
    expect(
      computeVisibleTagChipCount(
        [40, 40],
        TAG_CHIP_OVERFLOW_WIDTH_PX + TAG_CHIP_GAP_PX,
      ),
    ).toBe(0);
  });
});

describe("getDisplayConversationTags", () => {
  it("excludes reserved keys and puts priority keys first", () => {
    expect(
      getDisplayConversationTags({
        owner: "alice",
        acpserver: "claude-code",
        title: "can you research an office app?",
        origin: "slack",
        env: "prod",
        archiveworkspacepath: "/workspace/project",
        git_provider: "github",
        repo_name: "org/repo",
        selected_branch: "main",
        branch: "feature",
        repo: "other/repo",
        workspace: "/tmp/ws",
        working_dir: "/tmp/wd",
      }),
    ).toEqual([
      ["origin", "slack"],
      ["env", "prod"],
      ["owner", "alice"],
    ]);
  });

  it("ranks priority keys by their normalized name", () => {
    // The reserved-key filter normalizes the key, so the priority lookup must
    // too — otherwise a cloud-stamped `Origin` sorts alphabetically instead of
    // leading the row.
    expect(
      getDisplayConversationTags({
        env: "prod",
        Origin: "slack",
      }),
    ).toEqual([
      ["Origin", "slack"],
      ["env", "prod"],
    ]);
  });

  it("returns an empty list for nullish tags", () => {
    expect(getDisplayConversationTags(null)).toEqual([]);
    expect(getDisplayConversationTags(undefined)).toEqual([]);
  });

  it("drops blank and whitespace-only tag values", () => {
    expect(
      getDisplayConversationTags({
        appmode: "work",
        worktools: "",
        workwsid: "   ",
        owner: "alice",
      }),
    ).toEqual([
      ["appmode", "work"],
      ["owner", "alice"],
    ]);
  });
});

describe("getConversationTagLabelKind", () => {
  it.each([
    ["git_provider", "git"],
    // `origin` / `source` name where a conversation came from (Slack, an API
    // call, an automation), which is not a git fact — they stay "other" and
    // humanize to "Origin" / "Source".
    ["origin", "other"],
    ["source", "other"],
    ["repo_name", "repo"],
    ["selected_branch", "branch"],
    ["archiveworkspacepath", "workspace"],
    ["Appmode", "app_mode"],
    ["worktools", "work_tools"],
    ["Workwsid", "work_wsid"],
    ["owner", "other"],
  ] as const)("maps %s → %s", (key, kind) => {
    expect(getConversationTagLabelKind(key)).toBe(kind);
  });
});

describe("getConversationTagLabel", () => {
  const t = (key: I18nKey) => {
    switch (key) {
      case I18nKey.CONVERSATION_PANEL$PREVIEW_GIT:
        return "Git";
      case I18nKey.CONVERSATION_PANEL$PREVIEW_REPO:
        return "Repo";
      case I18nKey.CONVERSATION_PANEL$PREVIEW_BRANCH:
        return "Branch";
      case I18nKey.CONVERSATION_PANEL$PREVIEW_WORKSPACE:
        return "Workspace";
      case I18nKey.CONVERSATION_PANEL$PREVIEW_APP_MODE:
        return "App mode";
      case I18nKey.CONVERSATION_PANEL$PREVIEW_WORK_TOOLS:
        return "Work tools";
      case I18nKey.CONVERSATION_PANEL$PREVIEW_WORK_WSID:
        return "Workspace ID";
      default:
        return String(key);
    }
  };

  it("uses localized labels for known keys instead of wire names", () => {
    expect(getConversationTagLabel("selected_branch", t)).toBe("Branch");
    expect(getConversationTagLabel("repo_name", t)).toBe("Repo");
    expect(getConversationTagLabel("archiveworkspacepath", t)).toBe(
      "Workspace",
    );
    expect(getConversationTagLabel("Appmode", t)).toBe("App mode");
    expect(getConversationTagLabel("worktools", t)).toBe("Work tools");
    expect(getConversationTagLabel("Workwsid", t)).toBe("Workspace ID");
    expect(formatConversationTagTooltip("selected_branch", "main", t)).toBe(
      "Branch: main",
    );
  });

  it("humanizes unknown snake_case keys", () => {
    expect(humanizeConversationTagKey("my_custom_tag")).toBe("My custom tag");
    expect(getConversationTagLabel("owner", t)).toBe("Owner");
  });

  it("shows the label alone for bare tags (empty value)", () => {
    expect(formatConversationTagTooltip("work", "", t)).toBe("Work");
  });

  it("labels origin and source by their own names, not Git", () => {
    expect(getConversationTagLabel("origin", t)).toBe("Origin");
    expect(getConversationTagLabel("source", t)).toBe("Source");
    expect(formatConversationTagTooltip("origin", "slack", t)).toBe(
      "Origin: slack",
    );
    // The git host stamp keeps the Git label.
    expect(getConversationTagLabel("git_provider", t)).toBe("Git");
  });
});
