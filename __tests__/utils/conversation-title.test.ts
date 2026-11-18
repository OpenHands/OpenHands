import { describe, expect, it } from "vitest";
import {
  AUTOMATION_TITLE_SEPARATOR,
  AUTOMATION_WORK_ITEM_SEPARATOR,
  composeAutomationWorkItemTitle,
  getAutomationWorkItemOverrideTitle,
  getAutomationWorkItemTitleFromTags,
  isAutomationMachineTitle,
  isPlaceholderConversationTitle,
} from "#/utils/conversation-title";
import {
  WORK_ITEM_ID_TAG_KEY,
  WORK_ITEM_PROVIDER_TAG_KEY,
  WORK_ITEM_TITLE_TAG_KEY,
} from "#/utils/work-item-reference";
import { AUTOMATION_NAME_TAG_KEY } from "#/api/agent-server-adapter";

const workItemTags = {
  [WORK_ITEM_PROVIDER_TAG_KEY]: "github",
  [WORK_ITEM_ID_TAG_KEY]: "142",
  [WORK_ITEM_TITLE_TAG_KEY]: "Mission Provider Filtering",
};
const automationTags = {
  ...workItemTags,
  [AUTOMATION_NAME_TAG_KEY]: "\u{1F9EC} CIPHER",
};
const WORK_ITEM_NAME = "GH-142 \u2014 Mission Provider Filtering";
const COMPOSED = `\u{1F9EC} CIPHER${AUTOMATION_WORK_ITEM_SEPARATOR}${WORK_ITEM_NAME}`;

describe("automation work-item title composition", () => {
  it("prefixes the automation name to the work-item name", () => {
    expect(
      composeAutomationWorkItemTitle("\u{1F9EC} CIPHER", WORK_ITEM_NAME),
    ).toBe(COMPOSED);
  });

  it("returns null unless both halves are present", () => {
    expect(composeAutomationWorkItemTitle(null, WORK_ITEM_NAME)).toBeNull();
    expect(composeAutomationWorkItemTitle("\u{1F9EC} CIPHER", null)).toBeNull();
    expect(composeAutomationWorkItemTitle("   ", WORK_ITEM_NAME)).toBeNull();
    expect(composeAutomationWorkItemTitle("\u{1F9EC} CIPHER", "  ")).toBeNull();
  });

  it("collapses whitespace in the automation name but preserves the work item", () => {
    expect(
      composeAutomationWorkItemTitle("  \u{1F9EC}   CIPHER  ", WORK_ITEM_NAME),
    ).toBe(COMPOSED);
  });

  it("resolves the composed name from tags only when both tag sets are present", () => {
    expect(getAutomationWorkItemTitleFromTags(automationTags)).toBe(COMPOSED);
    expect(getAutomationWorkItemTitleFromTags(workItemTags)).toBeNull();
    expect(getAutomationWorkItemTitleFromTags(null)).toBeNull();
    expect(getAutomationWorkItemTitleFromTags({})).toBeNull();
  });
});

describe("automation machine title detection", () => {
  it("recognizes the SDK machine title and the bare automation name", () => {
    expect(
      isAutomationMachineTitle(
        `\u{1F9EC} CIPHER${AUTOMATION_TITLE_SEPARATOR}2026-11-18 19:21:42 UTC`,
        "\u{1F9EC} CIPHER",
      ),
    ).toBe(true);
    expect(
      isAutomationMachineTitle("\u{1F9EC} CIPHER", "\u{1F9EC} CIPHER"),
    ).toBe(true);
  });

  it("normalizes whitespace before comparing", () => {
    expect(
      isAutomationMachineTitle(
        `\u{1F9EC}  CIPHER ${AUTOMATION_TITLE_SEPARATOR} 2026-11-18`,
        " \u{1F9EC} CIPHER ",
      ),
    ).toBe(true);
  });

  it("does not treat a user rename, a blank title, or a missing name as a machine title", () => {
    expect(
      isAutomationMachineTitle("Renamed by the user", "\u{1F9EC} CIPHER"),
    ).toBe(false);
    expect(isAutomationMachineTitle("", "\u{1F9EC} CIPHER")).toBe(false);
    expect(
      isAutomationMachineTitle(
        `\u{1F9EC} CIPHER${AUTOMATION_TITLE_SEPARATOR}x`,
        null,
      ),
    ).toBe(false);
  });
});

describe("automation work-item display override", () => {
  it("replaces the SDK machine title with the composed name", () => {
    expect(
      getAutomationWorkItemOverrideTitle(
        automationTags,
        `\u{1F9EC} CIPHER${AUTOMATION_TITLE_SEPARATOR}2026-11-18 19:21:42 UTC`,
        "conv-1",
      ),
    ).toBe(COMPOSED);
  });

  it("replaces a placeholder or absent server title", () => {
    expect(
      getAutomationWorkItemOverrideTitle(automationTags, "", "conv-1"),
    ).toBe(COMPOSED);
    expect(
      getAutomationWorkItemOverrideTitle(
        automationTags,
        "Conversation conv-",
        "conv-1",
      ),
    ).toBe(COMPOSED);
  });

  it("yields to an explicit user rename", () => {
    expect(
      getAutomationWorkItemOverrideTitle(
        automationTags,
        "Renamed by the user",
        "conv-1",
      ),
    ).toBeNull();
  });

  it("returns null when the conversation is not automation-born or has no work item", () => {
    expect(
      getAutomationWorkItemOverrideTitle(workItemTags, "", "conv-1"),
    ).toBeNull();
    expect(
      getAutomationWorkItemOverrideTitle(automationTags, "", "conv-1"),
    ).not.toBeNull();
    expect(
      getAutomationWorkItemOverrideTitle(
        { [AUTOMATION_NAME_TAG_KEY]: "\u{1F9EC} CIPHER" },
        "",
        "conv-1",
      ),
    ).toBeNull();
  });
});

describe("isPlaceholderConversationTitle", () => {
  it("treats a blank or default title as a placeholder", () => {
    expect(isPlaceholderConversationTitle("", "abcde-123")).toBe(true);
    expect(isPlaceholderConversationTitle(null, "abcde-123")).toBe(true);
    expect(
      isPlaceholderConversationTitle("Conversation abcde", "abcde-123"),
    ).toBe(true);
    expect(isPlaceholderConversationTitle("Real title", "abcde-123")).toBe(
      false,
    );
  });
});
