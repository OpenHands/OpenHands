import { describe, expect, it } from "vitest";
import {
  computeVisibleTagChipCount,
  filterPreviewConversationTags,
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
  it("filters reserved keys and puts priority keys first", () => {
    expect(
      getDisplayConversationTags({
        owner: "alice",
        acpserver: "claude-code",
        origin: "slack",
        env: "prod",
        archiveworkspacepath: "/workspace/project",
        git_provider: "github",
        repo_name: "org/repo",
        selected_branch: "main",
      }),
    ).toEqual([
      ["origin", "slack"],
      ["git_provider", "github"],
      ["repo_name", "org/repo"],
      ["selected_branch", "main"],
      ["archiveworkspacepath", "/workspace/project"],
      ["env", "prod"],
      ["owner", "alice"],
    ]);
  });

  it("returns an empty list for nullish tags", () => {
    expect(getDisplayConversationTags(null)).toEqual([]);
    expect(getDisplayConversationTags(undefined)).toEqual([]);
  });
});

describe("getConversationTagLabelKind", () => {
  it.each([
    ["git_provider", "git"],
    ["repo_name", "repo"],
    ["selected_branch", "branch"],
    ["archiveworkspacepath", "workspace"],
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
    expect(formatConversationTagTooltip("selected_branch", "main", t)).toBe(
      "Branch: main",
    );
  });

  it("humanizes unknown snake_case keys", () => {
    expect(humanizeConversationTagKey("my_custom_tag")).toBe("My custom tag");
    expect(getConversationTagLabel("owner", t)).toBe("Owner");
  });
});

describe("filterPreviewConversationTags", () => {
  it("drops tags covered by existing hovercard fields", () => {
    expect(
      filterPreviewConversationTags(
        [
          ["git_provider", "github"],
          ["repo_name", "org/repo"],
          ["selected_branch", "main"],
          ["archiveworkspacepath", "/workspace/project"],
          ["owner", "alice"],
        ],
        {
          hasGitProvider: true,
          hasRepository: true,
          hasBranch: true,
          hasDirectory: true,
        },
      ),
    ).toEqual([["owner", "alice"]]);
  });
});
